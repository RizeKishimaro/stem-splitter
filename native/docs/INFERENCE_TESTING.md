# Inference testing

The tests load real ONNX graphs and compare the outputs with known
tensors. They do not stub the runtime. Python is used only to regenerate
the graphs; the test binary does not start it.

## Graphs

`native/inference/tools/make_test_model.py` writes two models with the
standard library (no `onnx` package):

- `testdata/scale_mul.onnx` — input `x` float32 `[1, 4]`, initializer
  `scale = 2`, output `y = x * 2`. The vector `1, 2, 3, 4` must come
  back as `2, 4, 6, 8`.
- `testdata/chain_mul.onnx` — 80 multiplies by `1` of a float32
  `[1, 1048576]` tensor. Loaded with `optimizeGraph = false` so the
  worker stays inside `Run` long enough for a second job to be queued
  and cancelled. The output equals the input.

Regenerate with:

```bash
python3 native/inference/tools/make_test_model.py
```

## Build on Linux

From `native/`, pointed at a local ONNX Runtime 1.20.1 tree (the
configure step downloads the official archive when this path is empty):

```bash
cmake -S . -B build/linux-inference -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSONICLAB_WITH_INFERENCE=ON \
  -DSONICLAB_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build/linux-inference --target soniclab_inference_tests soniclab_native_tests -j
```

The Linux x86_64 archive used for development is
`onnxruntime-linux-x64-1.20.1.tgz`, SHA256
`67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105`.
macOS archives are selected by CMake and have not been run.

## What is executed

```bash
./build/linux-inference/inference/soniclab_inference_tests inference.
./build/linux-inference/tests/soniclab_native_tests
```

`soniclab_inference_tests` is a separate binary so the Beat Analyzer
tests do not link ONNX Runtime. The inference binary's `BUILD_RPATH`
points at `libonnxruntime.so`. CTest sets `LD_LIBRARY_PATH` as well:

```bash
cd build/linux-inference && ctest -R 'inference|beat_analyzer' --output-on-failure
```

The inference group covers load, rejection of a garbage file, input
shape rejection, the numeric scale check, a second identical run, a
background job, cancellation of a job that is still queued behind the
chain model, unload, service destruction, the CPU provider, and a
missing provider name.

Address and thread sanitizers cannot be combined. Use a fresh build
directory for each:

```bash
cmake -S . -B build/linux-inference-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSONICLAB_SANITIZE_ADDRESS=ON \
  -DSONICLAB_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build/linux-inference-asan --target soniclab_inference_tests -j
./build/linux-inference-asan/inference/soniclab_inference_tests inference.

cmake -S . -B build/linux-inference-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSONICLAB_SANITIZE_THREAD=ON \
  -DSONICLAB_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build/linux-inference-tsan --target soniclab_inference_tests -j
./build/linux-inference-tsan/inference/soniclab_inference_tests inference.
```

AddressSanitizer reports two allocations inside `libonnxruntime.so`
(384 bytes, no SonicLab frame). `native/inference/lsan.supp` suppresses
that process-lifetime leak. CTest adds it when
`SONICLAB_SANITIZE_ADDRESS` is on. A direct run needs the same file:

```bash
LSAN_OPTIONS=suppressions=$PWD/inference/lsan.supp \
  ./build/linux-inference-asan/inference/soniclab_inference_tests inference.
```

The Beat Analyzer plugin tests stay in the existing JUCE build and do
not need this library:

```bash
./build/linux-juce/plugins/beat_analyzer/juce/SonicLabBeatAnalyzerPluginTests
```

These runs are Linux only. They do not validate Audio Units, Core ML,
or Metal.
