# Python ↔ native engine integration (Phase 3)

This note compares three ways for the Python app to drive a native C++
audio engine. It records the recommendation and describes the proof of
concept built under `native/engine_capi/` and `native/python/`. The Python
`AudioEngine` is still the engine the app uses. The prototype is not
imported anywhere in `soniclab/`, and a test enforces that.

Related documents:

* `docs/engine_interface.md`: the public surface the UI uses (Phase 2).
* `native/docs/ARCHITECTURE.md`: why the audio thread must leave Python (Phase 1).

## 1. Options compared

| | **1. pybind11 extension** | **2. C ABI + ctypes (or cffi)** | **3. Separate native process** |
|---|---|---|---|
| **Call cost** | About 100 ns. Zero-copy numpy through the buffer protocol | About 1 µs per call (ctypes argument conversion). Zero-copy numpy through `ndarray.ctypes` | 10–100 µs per message over a socket or pipe. Audio is zero-copy only through shared memory |
| **Realtime safety** | Safe only if bindings release the GIL and never call Python from the audio thread. pybind11 makes Python callbacks easy, so it takes discipline | ctypes `CDLL` releases the GIL on every call. The ABI can simply offer no callbacks, so the rule is structural | Audio runs in another process, so Python cannot block it at all. The UI ↔ engine latency is the cost |
| **Memory ownership** | Expressive (`py::array_t`, keep-alive policies), but lifetimes cross into the binding layer and need care | Explicit and narrow. Handles are created and destroyed in pairs, buffers are borrowed per call, results come back as copies. Easy to audit | Clearest: nothing is shared except named shared-memory regions with an agreed lifetime |
| **Complexity** | Medium. C++ templates, a wheel per Python version (pybind11 has no stable ABI), plus scikit-build-core or setuptools integration | Low. One shared library and a small `.py` wrapper. Signatures are written twice (C header and `argtypes`), and mistakes crash instead of raising | High. A protocol, process lifecycle and restart, shared-memory management, and state sync after a crash |
| **New dependencies** | pybind11 (not in `.venv`), plus a build backend | None: ctypes is stdlib, and numpy is already used. cffi is only in `.venv` as a sounddevice dependency | None, unless a serialization library is chosen |
| **Cross-platform** | Linux, macOS and Windows, but a compiled wheel per OS × arch × Python version | Linux, macOS and Windows. One library per OS × arch, independent of the Python version | Linux, macOS and Windows. Shared-memory and IPC APIs differ per OS |
| **Crash isolation** | None | None | Full. A crashing plugin takes down only the engine process |
| **Fits SonicLab** | Like `sounddevice` | Like `sounddevice` (which uses cffi over PortAudio's C API) | Like the Madmom and Demucs workers |

## 2. Recommendation

**Make a C ABI the engine boundary, and use ctypes as its first client.**
Keep the option to add a typed binding over the same ABI later, and to
move the whole engine into its own process later.

Reasons:

1. **Realtime safety by construction.** The ABI has no callback
   registration, and every function is GIL-free under `ctypes.CDLL`.
   Python therefore *cannot* run on the audio thread. That rule needs no
   review discipline to hold. pybind11 makes the opposite easy.
2. **Ownership you can audit in one header.** Every rule fits in
   `engine_capi.h`. It is enforced on both sides: the wrapper validates
   dtype, contiguity, writeability and shape, and the C side validates
   null pointers and ranges.
3. **No new dependencies, and no per-Python-version builds.** The same
   `.so`/`.dylib` works with Python 3.12 in `.venv` and with any later
   Python. The same library works from the Madmom worker's 3.11 venv too.
4. **It keeps the other options open.** A C ABI is exactly what a
   separate-process engine needs: serialize these same calls. It is also
   what a nanobind or pybind11 layer wraps if ctypes call overhead ever
   matters. With setters queued and meters polled at 30 Hz, it will not.
5. **The JUCE code stays behind the ABI.** Phase 4 can link JUCE into this
   library without anything in Python changing.

When to go further:

* **Out-of-process engine:** when in-process plugin crashes become a real
  support problem. Plugin *scanning* goes out of process from the start
  anyway (ARCHITECTURE.md §3).
* **Typed binding:** only if profiling shows ctypes overhead matters. It
  will not for UI-rate calls. Prefer nanobind, which has a stable ABI for
  3.12+.

## 3. What the prototype does

```
Python (any thread)                      native (libsoniclab_engine)
───────────────────                      ───────────────────────────
NativeEngine(...)            ──create──► sl_engine (owned by the wrapper)
set_output_gain/muted        ──queue───► SpscRing<EngineCommand> ─┐ (mutex among
                                                                  │  control threads only)
process(np.float32 array)    ──borrow──► processBlock in place    │
  (offline; BUSY while device runs)       ▲ drains commands ◄─────┘
start_device(tone)           ──────────► std::thread "device": sine → processBlock
                                           → SpscRing<float> output, AtomicPeak
read_output() / take_peak()  ◄──copy──── (mutex among control-thread readers only)
stats()                      ◄──copy──── atomics
close() / __exit__ / __del__ ─destroy──► stop + join device thread, free
```

* **Python → native:** queued commands (`set_output_gain`,
  `set_master_muted`) and the offline `process()` call.
* **Native processing:** gain, then mute, then clip to ±1, then peak. This
  is the master stage of `AudioEngine._callback`. The test
  `test_process_matches_python_master_stage` checks it is bit-identical to
  `render_clips` for gains 0–1.5 with and without mute.
* **No blocking on the audio side.** The audio-side owner (device thread
  or `process`) takes no lock, allocates nothing, and calls no client code.
  Ownership of the audio side is one `atomic<bool>` taken with `exchange`,
  never waited on. `test_device_thread_keeps_running_while_python_holds_the_gil`
  busy-loops in pure Python, holding the GIL, and checks that native
  blocks keep coming.
* **The stand-in device** is a paced `std::thread`, not real audio
  output. It has no realtime priority, so `max_block_nanoseconds` in the
  stats measures processing time, not scheduling jitter.

### Ownership and lifetime rules

| Object | Owner | Lifetime |
|---|---|---|
| `sl_engine*` handle | the Python `NativeEngine` | `create` → `close()`, called once. `__del__` and `__exit__` call `close()`; later calls raise `ValueError` |
| Device thread | the engine | `start_device` → `stop_device` or destroy (joined in both cases) |
| Buffer given to `process` | the caller (numpy) | Borrowed for the call only; never stored by native code |
| Arrays from `read_output` and `stats` | Python | Fresh copies; native memory is never exposed |
| Command and output rings | the engine | Allocated in `create`, freed in `destroy`; never resized |

## 4. Path to macOS AU plugins

The same layering carries over to AU without changing the Python side:

1. **Phase 4: a real device.** Replace the stand-in thread with a JUCE
   `AudioDeviceManager` callback inside `libsoniclab_engine`. On macOS
   this is CoreAudio; on Linux, ALSA or JACK. The C ABI is unchanged:
   `start_device` opens the device. The library links JUCE, and the
   licence questions in `LICENSING.md` then apply to it.
2. **Phase 5: the clip mixer** moves behind the ABI. Each track's buffer
   then runs through a `soniclab::host::PluginChain` (Phase 1) on the audio
   thread. New C functions load, move and bypass plugins and set
   parameters (by slot id). They use the chain's existing snapshot and
   queue mechanism, so the ABI stays non-blocking.
3. **AU hosting** comes from `JucePluginHost`, which already registers
   `AudioUnitPluginFormat` on macOS only. Scanning runs in a child process
   that writes the known-plugin list. Loading happens on the control
   thread, never on the audio thread.
4. **Plugin windows.** An AU editor needs an `NSView` and the JUCE message
   loop on the main thread. Qt also owns the main thread, so plugin
   editors either live in a native window opened by the library (JUCE's
   `DocumentWindow`, pumped from the Qt event loop) or run in the
   out-of-process engine with its own run loop. That is the strongest
   argument for option 3 once plugin UIs arrive, and the C ABI makes that
   move mechanical.
5. **SonicLab's own AU** (the Beat Analyzer) is independent of all this.
   It is a standalone `.component` built by `juce_add_plugin`, and it
   shares only `soniclab::common` and its analysis pipeline with the engine.

## 5. Limitations of the prototype

* Master stage only: no clips, no mixing, no loop or transport, no
  resampling. It does not implement `docs/engine_interface.md`.
* The device is simulated, with no audio output, no realtime thread
  priority and no xrun detection.
* Only two commands exist (output gain and master mute).
* `process()` is limited to `max_block_frames` per call. Callers split
  longer buffers.
* ctypes signatures are written twice (header and wrapper). A mismatch is
  caught only by the tests, not by a compiler.
* Built and tested on Linux only. The wrapper looks for `.dylib`/`.dll`
  on macOS and Windows, but has not been run there.
