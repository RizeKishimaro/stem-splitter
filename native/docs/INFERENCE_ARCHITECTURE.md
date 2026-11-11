# Inference architecture

SonicLab plugins that need a neural net call this library. They do not
include ONNX Runtime headers and they do not start Python. The Beat
Analyzer does not link this library; its tracker is unchanged.

## Layout

```
native/inference/include/soniclab/inference/   public headers
native/inference/src/                          tensor, session, worker
native/inference/testdata/                     generated graphs, not weights
native/inference/tools/make_test_model.py      regenerates those graphs
native/cmake/OnnxRuntime.cmake                 locates the shared library
```

`soniclab::inference` is a static library. ONNX Runtime is a private
imported shared library (`libonnxruntime.so`). A plugin that links
`soniclab::inference` must still be able to load that `.so` at runtime
(`BUILD_RPATH` in the test binary, or `LD_LIBRARY_PATH`, or an install
rpath when a plugin is packaged).

## Backend

ONNX Runtime 1.20.1 is the first backend. The public types are
`Tensor`, `Model`, `LoadOptions`, and `InferenceService`. Replacing the
runtime means rewriting `model.cpp` and keeping those types.

`Model::load` reads a model file the caller supplies. Nothing in the
repository is a trained checkpoint.

Each session is created with one intra-op thread and one inter-op thread.
The audio process should not grow an ONNX Runtime pool beside the audio
callback. With that setting, ThreadSanitizer reports no races in this
library. The official shared library was not built with the sanitizer;
leaving its default pool on produces reports whose writes are inside
`libonnxruntime.so`.

`LoadOptions` has two fields:

- `optimizeGraph` (default true). Graph rewrites stay on for production
  loads. The cancellation test turns them off so a chain of multiplies
  is not folded into nothing.
- `executionProvider` (default `CPUExecutionProvider`). The CPU provider
  is the session default; this library does not append it. Any other
  name is passed to `SessionOptions::AppendExecutionProvider` and the
  load fails with the runtime's error when that provider is absent.

On the Linux development machine the 1.20.1 x86_64 package exposes
`CPUExecutionProvider` and its headers contain no Core ML symbols.
`executionProviders()` lists whatever the loaded library reports.
Presence in that list is not evidence that a provider runs a model.
Core ML and Metal have not been tested, because this work was built on
Linux.

## Tensors

A `Tensor` is one contiguous `float` buffer plus a shape. The last
dimension changes fastest (row-major; NCHW when the model names its
axes that way). The tensor owns the buffer. `data()` is valid until
the tensor is moved-from or destroyed.

`assign` rejects a negative dimension and a value count that is not the
product of the shape. A model input whose declared dimension is
negative is dynamic: the rank must match and every static dimension
must match the tensor that is passed in. This milestone accepts
float32 only. Several inputs are allowed and are matched by position
to the model's input list, which is the shape a future stem model needs.
The test graph has one input, `x`, and one output, `y`.

`Model::run` reads the caller's floats and does not copy them. The
buffers must stay alive until `run` returns. `InferenceService::submit`
copies the tensors into the job so the caller may destroy them
immediately.

## Worker

`InferenceService` owns one thread named `soniclab-infer`. `submit`
enqueues a job and returns an id starting at 1. `status` reports
`Queued`, `Running`, `Succeeded`, `Failed`, or `Cancelled`. Progress is
0, 0.5, and 1 for those stages. ONNX Runtime does not report progress
inside a single `Run`, so there is no finer value.

`cancel` on a queued job means the worker never calls `Run`. `cancel`
on a running job sets a flag and, if `Run` has published its
`RunOptions`, calls `SetTerminate` from the cancelling thread. A run
that finishes before the terminate lands is still `Succeeded`.

`load` stores another owner of the model's session. `unload` rejects
later submits and leaves a job that was already accepted running,
because that job holds its own `shared_ptr` to the session. Destroying
the service sets the stop flag, asks in-flight runs to terminate,
wakes the worker, and joins it.

Do not call `load`, `run`, `submit`, or `wait` from `processBlock` or
any other audio callback. The worker uses a mutex and a condition
variable. That is acceptable because this thread is never the audio
thread. The Beat Analyzer's `AnalysisPipeline` stays lock-free and is
a different path.

## Adding a model

1. Keep the file outside the repository. Do not commit weights.
2. Export float32 inputs and outputs. Other element types fail at load.
3. Load it on a normal thread:

```cpp
soniclab::inference::LoadOptions options;
auto loaded = soniclab::inference::Model::load("/path/to/model.onnx", options);
if (!loaded.ok) { /* loaded.message */ }
```

4. Inspect `loaded.value.info()` for names and shapes before allocating
   tensors. A negative dimension is dynamic.
5. Give the model to a service and submit tensors in input order. Read
   the result from `wait` on a thread that is allowed to block.
6. To try a provider other than CPU, set `options.executionProvider`
   and treat a failed load as "this build cannot run that provider".
   Do not assume Core ML or Metal succeeded without a run on that Mac.
