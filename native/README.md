# SonicLab native (Phase 1)

C++17 groundwork for running SonicLab's audio in native code and for
hosting and building Audio Unit plugins with JUCE. This directory is
self-contained: it does not change the Python app, its playback, or its
project files.

* Architecture, the realtime problems in the current Python engine, and
  the Python ↔ native integration options: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Python ↔ native engine options and the Phase 3 prototype:
  [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md)
* JUCE, VST3 and Apple licensing: [LICENSING.md](LICENSING.md) — read
  before distributing anything built with JUCE.

## What is here

| Target | Needs JUCE | Contents |
|---|---|---|
| `soniclab::common` | no | lock-free SPSC ring, analysis markers, atomic peak meter, UI→audio commands |
| `soniclab::host_core` | no | plugin interfaces, lock-free plugin chain, chain state save/restore |
| `soniclab::analysis_core` | no | shared by all analysis plugins: audio thread → worker → UI pipeline, `OnlineAnalyzer` interface |
| `soniclab::beat_analyzer_core` | no | native beat tracker: tempo, beats, downbeats, beats per bar, live and whole-file |
| `soniclab_beat_cli` | no | development tool: the tracker on raw audio from stdin, used by `scripts/compare_beats.py` |
| `soniclab_native_tests` | no | 33 tests, dependency-free runner, CTest groups `common`, `host`, `analysis`, `beat_analyzer` |
| `soniclab::engine` (`libsoniclab_engine`) | no | Phase 3 prototype: C ABI shared library (`engine_capi.h`), master stage + stand-in device thread |
| `soniclab_engine_capi_tests` | no | 7 tests of the C ABI, CTest group `engine_capi` |
| `python/soniclab_native.py` | no | ctypes wrapper; 9 tests in `python/test_soniclab_native.py` |
| `soniclab::host_juce` | yes | JUCE scanner/loader/instance adapter (AU on macOS, VST3 opt-in) |
| `SonicLabBeatAnalyzer` | yes | JUCE plugin (Phase 4): pass-through, mono/stereo, Enabled parameter with saved state, input level meter, analysis FIFO, placeholder marker strip, drag-and-drop audio file analysis (waveform, length, peak). AU on macOS, Standalone everywhere |
| `SonicLabBeatAnalyzerPluginTests` | yes | 23 headless tests of the real processor, editor, file analysis and beat tracking (CTest `beat_analyzer_plugin`), plus an opt-in test that runs a real recording through the plugin (`SONICLAB_REAL_AUDIO`, used by `scripts/compare_beats.py --plugin`) |
| `plugins/shared/juce` + `cmake/SonicLabPlugin.cmake` | yes | what every SonicLab plugin shares: theme, widgets, processor and editor base classes, dropped-file analysis, and `soniclab_add_analysis_plugin()` |

## CMake options

| Option | Default | Meaning |
|---|---|---|
| `SONICLAB_BUILD_TESTS` | ON | Build the unit tests |
| `SONICLAB_BUILD_ENGINE_CAPI` | ON | Build `libsoniclab_engine` (Phase 3 prototype) |
| `SONICLAB_PYTHON` | empty | Python with numpy and `soniclab` installed; adds the ctypes tests to CTest as `python_ctypes` |
| `SONICLAB_WITH_JUCE` | OFF | Add the JUCE host adapter and the plugin |
| `SONICLAB_JUCE_DIR` | empty | Path to a JUCE source tree (checkout or submodule) |
| `SONICLAB_FETCH_JUCE` | OFF | Allow FetchContent to download `SONICLAB_JUCE_TAG` |
| `SONICLAB_JUCE_TAG` | `8.0.4` | JUCE release tag to fetch |
| `SONICLAB_BUILD_VST3` | OFF | Also build/host VST3 (see LICENSING.md) |
| `SONICLAB_SANITIZE_THREAD` / `_ADDRESS` | OFF | TSan, or ASan + UBSan |

JUCE is found in this order: `SONICLAB_JUCE_DIR`, then an installed JUCE
via `find_package(JUCE)`, then FetchContent only if `SONICLAB_FETCH_JUCE=ON`.
With none of those, configuring with `SONICLAB_WITH_JUCE=ON` stops with
a message saying what to set. Nothing is downloaded by default.

Presets need CMake 3.25 or newer (the project itself needs 3.22).

## Arch Linux

Shared code and tests (no JUCE, no download):

```bash
sudo pacman -S --needed base-devel cmake ninja
cd native
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

Python ↔ native prototype (needs the build above; the app does not use it):

```bash
cd ..   # repo root
.venv/bin/python -m unittest discover -s native/python -v
```

The wrapper finds `native/build/<preset>/engine_capi/libsoniclab_engine.so`
automatically, or the path in `SONICLAB_NATIVE_LIB`. To run it under CTest
instead, configure with `-DSONICLAB_PYTHON="$PWD/../.venv/bin/python"`.

Also available: `linux-release`, `linux-tsan` (ThreadSanitizer) and
`linux-asan` (Address + UB sanitizers), each with the same three commands.
Clang works too: `CXX=clang++ cmake -S . -B build/clang -G Ninja`.

With JUCE (Standalone build of the plugin; AU is never built on Linux):

```bash
sudo pacman -S --needed alsa-lib freetype2 fontconfig libx11 libxrandr libxinerama libxcursor libxext libxcomposite mesa
scripts/get_juce.sh
cmake --preset linux-juce
cmake --build --preset linux-juce
ctest --preset linux-juce
```

`scripts/get_juce.sh` clones JUCE 8.0.4 into `third_party/JUCE` (gitignored)
and checks the commit hash; the `linux-juce` and `macos-xcode` presets look
there. `-DSONICLAB_FETCH_JUCE=ON` fetches with CMake instead. `ctest` then
runs the shared-code tests plus `beat_analyzer_plugin`, which drives the
real `BeatAnalyzerProcessor` headless. The Standalone app is at
`build/linux-juce/plugins/beat_analyzer/juce/SonicLabBeatAnalyzer_artefacts/Debug/Standalone/`. Add
`-DSONICLAB_BUILD_VST3=ON` for a VST3 for local testing; read the VST3
section of LICENSING.md before giving it to anyone. JACK support needs
`jack2` or `pipewire-jack`.

## macOS: AUv2 build for Logic Pro

Step-by-step guide for whoever builds on the Mac, including validation,
Logic Pro scanning and troubleshooting: **[docs/MACOS_AU_BUILD.md](docs/MACOS_AU_BUILD.md)**.

Short version (macOS 11+, full Xcode opened once, CMake 3.25+, git):

```bash
cd native
scripts/build_au_macos.sh --install
```

That clones the pinned JUCE, configures the `macos-xcode` preset (Xcode
generator, universal arm64 + x86_64), builds `SonicLabBeatAnalyzer_AU`,
ad-hoc signs `SonicLab Beat Analyzer.component`, copies it to
`~/Library/Audio/Plug-Ins/Components/`, refreshes the AU cache and runs
`auval -v aufx Btan Snlb`.

The shared code and tests also build on macOS without JUCE:
`cmake --preset macos-ninja && cmake --build --preset macos-ninja && ctest --preset macos-ninja`.

## Status and what is not done

Verified on Arch Linux (Phase 4): JUCE 8.0.4 configures; the host adapter,
the plugin's Standalone build and the plugin tests compile with no
warnings; the 23 plugin tests pass (bit-exact pass-through for mono and
stereo, no heap allocation in `processBlock`, metering, state round trip
and older/newer states, headless editor paint, dropped-file analysis of WAV
files, unreadable files, cancelling, tempo/beats/downbeats of a dropped
groove, live tempo through `processBlock`).

Beat tracker accuracy, measured with `scripts/compare_beats.py` against
the DAW's Madmom analysis on ten full songs (pop, house, rock, Vocaloid):
tempo within 0.2 BPM on 8 (one read double, one 2/3), mean beat F-measure
0.81 and mean downbeat F-measure 0.59 at ±70 ms. Six songs score above
0.9 on both. Typical misses: half-bar downbeats, and beats on dense fast
material. This is a classical tracker, not a neural one; Madmom stays the
DAW's reference.

The same ten songs through the plugin itself (`scripts/compare_beats.py
--plugin`: dropped as a file, and played live through `processBlock` on a
stereo and on a mono bus): file results are identical to the above. Live,
scored after the first 8 s, beat F-measure is 0.73 and downbeat 0.52;
mono and stereo give the same result, the audio passed through
bit-exact, and no allocation or dropped audio occurred on the audio
thread. The live tempo readout at the end of each song matched Madmom on
5 of 10: on three songs it read "no tempo" because their last seconds
fade out, and two are the same half/two-thirds cases as the file.
Live is weaker than file analysis because it sees 12 s at a time.
The Standalone app has been opened and used on Arch Linux (Wayland,
PipeWire). A one-off local VST3 build was also loaded through
SonicLab's own `JucePluginHost` and `PluginChain`. Scan, load, bit-exact
processing, a queued parameter change, and chain save/restore all worked.
That check found and fixed two Phase 1 bugs:

* JUCE needs the C language enabled.
* A restored VST3 chain could not find its plugin, because of a mismatched
  plugin identifier.

Not verified anywhere yet:

* **The AU itself.** It is built only on macOS. Version 0.1.x was built
  and installed on a Mac by the project owner; no `auval` or Logic Pro
  results have been recorded here, and 0.2.0 (beat tracking, new
  parameters) has not been built on a Mac. See the checklist in
  docs/MACOS_AU_BUILD.md.
* The beat tracker on live audio in a real host, and on Apple Silicon.
* Dragging files onto the plugin window inside Logic Pro. Logic may handle
  drags from its own browser itself; drags from Finder should reach the
  plugin.

Not done:

* The beat tracker does not follow tempo changes inside a file (one tempo
  per file; the live mode re-estimates every half second), and chooses
  only between 3 and 4 beats per bar unless the bar length is set by hand.
* Onset Detector and Alignment Tool plugins: the shared base is ready,
  the plugins are not written.
* No native engine beyond the Phase 3 master-stage prototype, no real
  device I/O, and no connection to the Python app (see
  docs/ENGINE_INTEGRATION.md §5).
* No out-of-process plugin scanner, AUv3, Developer ID signing,
  notarization or installer.
* Plugin-chain parameter changes queued for a slot that the audio thread
  has not picked up yet are dropped; a later phase can re-queue them.
