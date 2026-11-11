# SonicLab native architecture (Phase 1)

This note records what the Python application does today, the options for
bringing a native C++ engine in beside it, the realtime problems a native
engine has to fix, and the structure Phase 1 sets up under `native/`.
Nothing under `soniclab/` changes in Phase 1.

## 1. What exists today

```
 PySide6 UI (main_window.py, timeline.py, mixer.py, ...)
   │  QTimer every 30 ms: position, take_peaks(), read_master()/read_track()
   │  edits: set_clip_gain/pan/muted/solo/start/region, seek, play, loop
   ▼
 AudioEngine (soniclab/core/audio_engine.py)
   │  threading.RLock shared by every public method AND the callback
   │  _clips: list of dicts {audio (N×2 float32 @ 44.1 kHz), start, offset,
   │          length, fade_in, fade_out, clip_gain, gain, pan, muted, solo}
   │  clip index == project track index
   ▼
 sounddevice.OutputStream(44100 Hz, 2 ch, float32, blocksize 512)
   └─ _callback: for each clip, numpy slice × gain/pan × fade envelope,
      sum, output gain, clip to ±1, per-track and master peaks,
      copy into StereoRing scope buffers, advance position (with loop wrap)

 render_clips()   offline twin of _callback, used by bar_export.py so an
                  exported bar range matches playback sample for sample
 Recorder         sounddevice.InputStream, separate from playback
 Analysis         beats.py (aubio, in-process, off the callback)
                  beat_analysis.py → tools/madmom_worker.py in .venv-madmom
                  (subprocess, JSON over stdout, content-hash cache)
                  beat_jobs.py / onset_jobs.py: one worker thread each,
                  results polled by the UI
                  stems.py: Demucs in a subprocess
 Bars             bar_map.py: bar lines in seconds; bar_export.py rounds
                  bar edges to samples once
 Project          project.py: JSON {version, name, bpm, time_signature,
                  bars, tracks[]}; tracks carry gain/pan/mute/solo/region
```

Observations that matter for a native engine:

* **There is no plugin or insert concept yet.** Processing per track is
  gain × pan × fade only. A plugin chain is new behaviour, not a port.
* **The UI reaches into engine internals**: `main_window.py` reads
  `engine._clips`, `engine._output_gain` and `engine._master_muted`
  directly. A swap-in native engine needs those exposed as public API first.
  That is a change to `main_window.py` and belongs to Phase 2.
* **Offline render must stay bit-identical to playback.** Today that is
  achieved by duplicating the mix code in `render_clips` and `_callback`. A
  native engine should have one mix function used by both the device
  callback and offline export.
* **Long-running work already lives in other processes** (Madmom, Demucs),
  with a JSON contract and cancel/timeout handling. That pattern is a good
  fit for plugin scanning.

## 2. Realtime-safety problems in the current engine

These are why the audio thread should move to native code, not just why
it would be faster.

| # | Where | Problem | Native fix |
|---|---|---|---|
| 1 | `_callback` takes `self._lock` (RLock) | Every UI setter, `position`, `take_peaks()` and `read_track()` take the same lock. A UI thread holding it (or descheduled while holding it) stalls the audio callback: priority inversion. | UI → audio commands through `SpscRing<EngineCommand>` (`engine_command.h`), applied at block start. No lock on the audio thread. |
| 2 | The callback is Python | It needs the GIL. Any thread holding the GIL (Qt painting, numpy work that does not release it, aubio, JSON parsing) delays the callback. GC pauses land inside it. | The native callback never touches Python. Python talks to the engine only from its own threads. |
| 3 | Allocation in the callback | `list(self._clips)`, numpy slices and products (`audio[...] * left_gain`), `np.arange`/`np.ones` fade envelopes, the `segments` list. | Preallocated buffers sized at prepare time; the mix loop is plain C++ over raw pointers. |
| 4 | `StereoRing.push` locks | The scope rings take their own `threading.Lock` in the callback while the UI copies out under it. | `SpscRing<float>` per scope: the audio thread pushes, the UI pops; neither blocks. |
| 5 | `play()` opens the stream under the lock | PortAudio start-up runs while the lock is held, so the first callback can wait on it. | Device start/stop happen on the message thread with no lock the callback uses. |
| 6 | Peaks reset under the lock | `take_peaks()` resets while the callback writes. | `AtomicPeak`: CAS-max on the audio thread, `exchange(0)` on the UI. |
| 7 | Clip list replaced under the lock | `replace_clips` swaps the list; fine with the lock, unsafe without. | Same publish/retire snapshot scheme as `PluginChain` for the clip list. |

## 3. How a native engine could plug into Python

| | **A. nanobind / pybind11 extension** | **B. ctypes over a C ABI** | **C. Separate process + shared memory/IPC** |
|---|---|---|---|
| Call overhead | Lowest; zero-copy numpy views (`nb::ndarray`) | Low; manual pointer marshalling | Highest per call; audio in shared memory is zero-copy, commands go over a socket/pipe |
| Build | CMake + scikit-build-core produce a wheel per Python version (nanobind stable ABI covers 3.12+) | Plain shared library; Python side needs no compiler | Plain executable; Python needs no compiler |
| Type safety | Good, generated bindings | None; mistakes are crashes | Defined by the protocol |
| Crash isolation | None: a plugin crash kills SonicLab | None | Full: a plugin crash kills the engine process, the UI survives and can restart it |
| GIL | Release it in every binding that can block | ctypes releases it per call | Not involved |
| Debugging | One process | One process | Two processes, more moving parts |
| Fits existing code | Like `sounddevice`, an in-process module | Same | Like the Madmom and Demucs workers |

**Recommendation.** Define the engine boundary as a small **C ABI**
(`soniclab_engine.h`: create/destroy, load clip from a float buffer, queue
command, read meters/scope, render offline). Bind it to Python with
**nanobind** for day-to-day use (B stays available as a fallback and for
tests). Run the engine **in-process first**: it is the smallest step and
keeps offline export simple. Run **plugin scanning out-of-process from the
start** (a child process using `JucePluginHost::scan`, results written to
the known-plugins cache) because scanning is where third-party plugins most
often crash. If in-process plugin crashes become a real problem, the same C
ABI commands can be serialized to move the whole engine to a separate
process (option C) without changing the Python side.

## 4. Phase 1 structure

```
native/
  CMakeLists.txt, CMakePresets.json, cmake/SonicLabJuce.cmake
  common/                  soniclab::common (no JUCE)
    spsc_ring.h              lock-free SPSC ring, power-of-two, 64-bit indices
    analysis_marker.h/.cpp   onset/beat/downbeat/bar-line markers in samples
    atomic_peak.h            lock-free peak meter
    engine_command.h         POD UI→audio commands
  plugin_host/             soniclab::host_core (no JUCE)
    plugin_description.h     JUCE-free mirror of juce::PluginDescription
    plugin_instance.h        IPluginInstance: prepare/process/params/state
    plugin_scanner.h         IPluginScanner (discovery), IPluginLoader (load)
    plugin_chain.h/.cpp      insert chain: snapshot publish/retire, queued params
    chain_state.h/.cpp       capture/encode/decode/restore chain state
    juce_plugin_host.h/.cpp  JUCE adapter (only with SONICLAB_WITH_JUCE)
    juce/CMakeLists.txt      soniclab::host_juce: AU on macOS, VST3 opt-in
  plugins/shared/          what every SonicLab analysis plugin shares
    analysis_pipeline.h/.cpp soniclab::analysis_core (no JUCE):
                             audio thread → chunk FIFO → worker → marker FIFO,
                             OnlineAnalyzer interface (reset/analyze/finish/summary)
    juce/Theme.h             colours, fonts and LookAndFeel from the Python app
    juce/Widgets.*           level meter, LCD frame, confidence bar
    juce/AnalysisViews.*     dropped-file waveform view, live marker lane
    juce/FileAnalysis.*      background analysis of a dropped audio file
    juce/AnalysisProcessorBase.*  pass-through, metering, parameters, versioned
                             state, pipeline, file analysis
    juce/AnalysisEditorBase.*     header, footer, Enabled, meter hints, drag and drop
  cmake/SonicLabPlugin.cmake  soniclab_add_analysis_plugin(): one call per plugin
  plugins/beat_analyzer/
    beat_tracker.h/.cpp      soniclab::beat_analyzer_core (no JUCE): onset
                             features, tempo, DP beat tracking, downbeats, meter
    src/PluginProcessor.*    parameters (tempo range, beats per bar) + tracker
    src/PluginEditor.*       tempo LCD, settings row
    juce/CMakeLists.txt      soniclab_add_analysis_plugin + headless tests
  tools/beat_cli.cpp       tracker on raw audio, for scripts/compare_beats.py
  tests/                   dependency-free runner, 33 tests, one CTest per group
  docs/ARCHITECTURE.md     this file
```

Why the JUCE-free split: everything with threading logic (ring, chain
snapshot swap, analysis pipeline) is unit-tested on every build, including
under ThreadSanitizer, without downloading JUCE. The JUCE files are thin
adapters.

### Plugin chain threading

```
message thread                        audio thread
--------------                        ------------
insert/remove/move/bypass
  → build Snapshot (new)              process():
  → pending_.exchange(snapshot) ───►    next = pending_.exchange(nullptr)
     (frees an unclaimed one)            if next: retired_.push(current_); current_ = next
queueParameter(slotId, i, v) ──SPSC──►  apply queued parameter changes by slot id
collectGarbage() ◄──────────SPSC─────   (old snapshot handed back, never freed here)
  → delete old snapshot, which may
    drop the last ref to a removed plugin
```

The audio thread swaps only when the retire ring has room, so it never
frees memory. Plugins are prepared before they are published and released
or destroyed on the message thread.

### Beat Analyzer threading

```
processBlock (audio)                 worker thread                 editor Timer (UI, 30 Hz)
  audio untouched                      pop AudioChunk                collectMarkers()
  mono downmix in 256-frame chunks     analyzer->analyze(chunk)      repaint marker strip
  stamped with stream position  ─FIFO► emit markers ─────FIFO──────►
  FIFO full → drop + count
```

The worker polls every 5 ms instead of waiting on a condition variable,
because notifying one from the audio thread would need a mutex.

## 5. Phase 2 and later (not done)

1. Add public accessors to `AudioEngine` for what the UI reads privately,
   then a `NativeAudioEngine` Python class with the same methods, selected
   by an environment variable, so both engines run side by side.
2. Native mix core with one function for playback and offline render;
   golden tests comparing it to `render_clips` on the same inputs.
3. JUCE `AudioDeviceManager` for output and recording.
4. Per-track `PluginChain` in the native mix; out-of-process scanner.
5. Chain state stored in the project JSON as a new optional per-track key,
   with a project `version` bump and loader fallback.
6. The real beat detector behind `OnlineAnalyzer`.
