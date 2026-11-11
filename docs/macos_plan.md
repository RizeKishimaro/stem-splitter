# SonicLab on macOS: plugin family and 6-stem splitter — implementation plan

Status: Phases A and B are implemented and tested on Linux (see
`native/README.md` for what was measured); they have not been built on a
Mac yet. Phase C (Stem Splitter panel) and the model, device and caching
parts of Phase D are implemented and tested on Linux against a stand-in
for Demucs (see `docs/stem_splitter.md`); no real Demucs run has been made
with them yet. The rest of D, and E to G, are plan only.
Product target is macOS (Apple Silicon first, Intel where dependencies
allow). The Linux machine stays the development and test host because it
is the only machine the developer has; every Mac-only claim needs a run
on a real Mac.

## 1. What exists today

| Area | State | Where |
|---|---|---|
| Python DAW (PySide6) | **exists**. Timeline, mixer, inspector, recorder, theme | `soniclab/ui/`, `soniclab/core/` |
| Stem splitting | **exists**, 4 stems: `htdemucs_ft` via `python -m demucs` in a separate venv, progress parsing, cancel, cache by SHA-256, import as tracks, banner UI | `soniclab/core/stems.py`, `main_window.py` (`StemJobQueue`, `stem_banner`) |
| Beat / downbeat analysis | **exists**: Madmom in an isolated Python 3.11 worker process, cached; aubio beats and onsets in-process | `soniclab/core/beat_analysis.py`, `tools/madmom_worker.py`, `beats.py`, `onsets.py` |
| Drum track pick, alignment, bar map | **exists** | `drums.py`, `alignment.py`, `bar_map.py`, `downbeat_*.py` |
| Native layer | **exists**: lock-free common code, plugin host interfaces, C ABI engine prototype, Python ctypes wrapper | `native/` |
| Beat Analyzer plugin | **exists**, scaffold only: pass-through, level meter, Enabled parameter, saved state, dropped-file waveform, SonicLab theme. **No beat detection** | `native/plugins/beat_analyzer/` |
| AUv2 on a Mac | built and installed by the developer's boss; `auval` / Logic results not yet reported back | `native/docs/MACOS_AU_BUILD.md` |

## 2. Findings that shape the plan

1. **Madmom and aubio cannot go into the plugins.** Madmom is Python with
   CC BY-NC-SA models; aubio is GPLv3. A native beat tracker has to be
   written in C++ for the plugins. The DAW keeps using Madmom in its
   worker, so the app and plugin can disagree; the plugin shows its own
   confidence and the DAW remains the reference.
2. **`htdemucs_6s` piano is weak.** The Demucs README says the piano
   source "is not working great at the moment." Piano will be exported,
   but quality has to be measured and stated, not promised.
3. **Demucs is in maintenance only.** The maintained fork
   (github.com/adefossez/demucs) says no new features are planned. Code is
   MIT; the README says the **weights' licence is not clear** (MusDB is
   research-only). Weights must stay a separate, user-initiated download,
   and commercial distribution needs a licence decision.
4. **Apple GPU is not a given.** The Demucs README documents only CUDA and
   CPU. PyTorch MPS has known problems with Demucs (complex STFT ops), and
   Intel Macs are stuck at PyTorch 2.2 / Python ≤ 3.12. Third-party MLX and
   Core ML ports of `htdemucs_6s` exist and claim large speed-ups, but are
   unofficial. Plan: CPU everywhere first, then measure MPS (with
   `PYTORCH_ENABLE_MPS_FALLBACK=1`) and an MLX port on a real Mac; offer
   GPU only for what passes a numerical-equality and speed check.
5. **Snare: no commercially usable model found yet.** LarsNet separates a
   drum stem into kick, snare, toms, hi-hat and cymbals, but its weights
   are **CC BY-NC 4.0** (non-commercial) and its code has no licence file.
   DSP-only methods (NMF with drum templates, transient/HPSS masking) can
   emphasise snare but are not true separation.

## 3. Architecture

```
SonicLab.app (Python + PySide6)
├─ Stem Splitter panel (new) ── StemJobQueue ── separation worker process
│                                                 ├─ demucs (htdemucs_6s, htdemucs_ft)
│                                                 └─ optional drum splitter (snare)
├─ Madmom worker process (exists)
└─ timeline / mixer / import (exists)

~/Library/Application Support/SonicLab/
├─ engines/demucs-venv/      created by the app on first use, not bundled
└─ models/                   Demucs / drum weights, downloaded on request

SonicLab plugin family (C++ / JUCE, AUv2)
native/plugins/
├─ shared/  soniclab_plugin_ui   theme, header, meter, section, LCD, file panel,
│                                marker lane, buttons, toggle, combo, slider
│           soniclab_plugin_base AnalysisPluginProcessor: APVTS, versioned state,
│                                pass-through bus logic, input metering,
│                                AnalysisPipeline + FileAnalysis
├─ beat_analyzer/   BeatTracker (onset flux, tempo, DP beats, downbeats, meter)
├─ onset_detector/  (later) reuses base + UI, onset lane
└─ alignment_tool/  (later) reuses base + UI, two-input comparison
```

Rules carried over: nothing blocks the audio thread; plugins never
download or load models; analysis runs on worker threads; the DAW runs
inference in separate processes.

## 4. Phases

Each phase ends with tests on Linux and a Mac checklist. "Mac" items can
only be ticked from a real Mac report.

### Phase A — shared plugin UI and base (native) — done on Linux
* Move `Theme.h`, `LevelMeter`, header/footer/section/LCD drawing and the
  dropped-file panel into a `soniclab_plugin_ui` static library; add
  themed slider, combo box and segmented control (needed for settings).
* `AnalysisPluginProcessor` base: bus layouts (mono/stereo pass-through),
  metering, APVTS with versioned state, `prepareToPlay` generation logic,
  FileAnalysis. Beat Analyzer becomes the first subclass.
* `juce_add_plugin` helper with shared metadata (manufacturer `Snlb`,
  bundle id prefix `org.soniclab.`), so new plugins get identical setup.
* Tests: existing 16 plugin tests keep passing; UI components render
  headless; a second dummy plugin builds from the template.

### Phase B — Beat Analyzer detection (native) — done on Linux
* Onset strength: log-magnitude spectral flux at ~100 fps on a mono
  downmix, resampled to 22.05 kHz on the worker thread.
* Tempo: autocorrelation of the onset envelope weighted by a log-Gaussian
  tempo prior (range setting, default 60–200 BPM), with octave
  correction; confidence = peak prominence.
* Beats: dynamic-programming beat tracking over the envelope
  (Ellis 2007), online with a few seconds of look-ahead for live input,
  full-file for dropped files.
* Downbeats and meter: score bar phase and 3/4 vs 4/4 by low-band energy
  and onset accent per beat position; report meter only above a
  confidence threshold, otherwise "unknown".
* Parameters (automatable, saved): Enabled, Tempo range min/max, Meter
  (Auto / 3 / 4 / 6 / 7), Sensitivity, Hold tempo. Read-only state saved:
  last BPM. Visuals: BPM and meter readout with confidence bars, beat /
  downbeat lane (live) and over the dropped-file waveform, "Analyse
  again" and "Copy BPM".
* Validation: `native/scripts/compare_beats.py` runs the tracker
  (`soniclab_beat_cli`) and Madmom on the same files. Result on ten songs:
  tempo right on 8, mean beat F 0.81, mean downbeat F 0.59.
* Not built from the original Phase B list: Sensitivity and Hold tempo
  parameters, Copy BPM, and meters other than 3 and 4 in Auto.

### Phase C — Stem Splitter panel (Python DAW) — done, not yet run with real Demucs
* New dock / tab "Stem Splitter": file picker + drag-drop, waveform
  preview, model (`htdemucs_6s` 6 stems, `htdemucs_ft` 4 stems), device
  (Auto / CPU / Apple GPU — only shown once verified), output folder,
  per-stem checkboxes, Split / Cancel, progress per model, status.
* After splitting: per-stem play / solo preview through the existing
  engine, Export selected, Export all, Add to project as tracks (aligned
  at the source clip's start).
* Generalise `stems.py`: model table (name → stems), `STEMS` per model,
  `--device`, `--segment` for long files and low memory, `--shifts`,
  cache key includes model; current 4-stem import-on-split keeps working.

### Phase D — 6-stem processing and engine management
* Separation runs in its own venv created by the app (`python -m venv`,
  `pip install demucs` with a pinned PyTorch), with a "Install separation
  engine" step showing size and licence notes; weights downloaded on the
  first split and cached under Application Support.
* Long recordings: process in segments; memory guard; disk-space check.
* Tests on Linux with real audio (CPU): stem count, length equality with
  the source (sample-exact), sum-of-stems vs mix error, cancellation
  leaves no partial stems, cache reuse.

### Phase E — Snare extraction (research gate)
* Evaluate on drum stems from Phase D: LarsNet (quality reference,
  non-commercial), NMF-D with snare templates, HPSS + transient masking.
* Only ship a method that clearly isolates snare on test songs; label it
  "Snare (experimental)" in a separate "Drum parts" group, never in the
  Demucs group; keep it sample-aligned with the source.
* Needs a decision (below) before LarsNet is offered at all.

### Phase F — DAW integration
* Stems land as a track group named after the song, aligned, coloured per
  stem; project file stores the stem set and model so a reopened project
  finds them; undo works; Madmom downbeats can be taken from the drums
  stem automatically (reuses `drums.py`).

### Phase G — macOS packaging, signing, validation
* App: PyInstaller (or Briefcase) `.app`, arm64 and x86_64 builds
  (universal2 Python bundles are impractical with numpy / scipy / PySide6
  wheels), bundle id `org.soniclab.app`, Info.plist, icon.
* AU: universal `.component` (exists), version in sync with the app.
* `scripts/macos/sign_and_notarize.sh`: Developer ID Application signing
  with hardened runtime and entitlements, `notarytool submit --wait`,
  `stapler staple`; `productbuild` installer that puts the app in
  /Applications and the AU in /Library/Audio/Plug-Ins/Components; DMG.
* Validation checklist: `auval`, Logic scan, automation, state, mono and
  stereo, separation on CPU and Apple GPU, cancellation, import alignment.

## 5. Decisions needed from the product owner

1. **Snare model licence.** LarsNet weights are CC BY-NC 4.0. Options:
   offer it as an optional download marked non-commercial; use a DSP
   method only; or skip snare until a commercially licensed model is found.
2. **Demucs weights licence.** The weights' terms are unclear. Fine for
   internal use; for selling SonicLab, get legal advice or train/licence
   other weights. The app will download them on request, never bundle them.
3. **JUCE licence.** Distributing closed-source plugins needs a commercial
   JUCE licence (see `native/LICENSING.md`).
4. **Apple Developer account.** Signing and notarization need a Developer
   ID (team ID, certificates, an app-specific password or API key for
   `notarytool`).
5. **Order.** Proposed: A → B (testable here) while the Mac reports on the
   current AU; then C → D; E as a research spike; F; G.

## 6. What can be verified where

| Check | Linux (here) | Mac (needed) |
|---|---|---|
| Plugin code, UI rendering, beat tracker accuracy, state | yes | — |
| AU build, `auval`, Logic scan, automation | no | yes |
| Demucs 6s on CPU, cancellation, alignment, import | yes (after installing the engine, several GB) | yes |
| Apple GPU (MPS / MLX) speed and correctness | no | yes |
| Signing, notarization, installer | no | yes |

## Sources

* Demucs (maintained fork): https://github.com/adefossez/demucs
* LarsNet: https://github.com/polimi-ispl/larsnet and https://arxiv.org/abs/2312.09663
* demucs-mlx (unofficial MLX port): https://pypi.org/project/demucs-mlx/
* HTDemucs 6s Core ML (unofficial): https://huggingface.co/schism-audio/htdemucs-6s-coreml
