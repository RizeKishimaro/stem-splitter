# Stem Splitter

A dock beside the Mixer (View → Stem Splitter, or File → Stem Splitter…)
that splits one song into stems with Demucs, lets you listen to them, and
exports them or adds them to the project as tracks.

Status: built and tested on Linux with a stand-in for Demucs (32 tests).
**No real Demucs run has been made through it yet**, on any platform, so
stem quality, speed, memory use and the Apple GPU are all unmeasured.

## Using it

1. **Song.** Choose File…, or drop an audio file on the panel (a file
   dropped anywhere else in the window is imported into the project as
   before). The ▶ button plays the song.
2. **Split.** Pick the model and device, then Split. Progress and elapsed
   time show under the bar; Cancel stops Demucs and saves nothing.
3. **Stems.** Each stem has a ▶ button to listen to it. Tick the stems you
   want, then:
   * **Export Selected… / Export All…** copies them to a folder as
     `<song> - <stem>.wav`. Existing files are never overwritten.
   * **Add to Project** adds one track per ticked stem, all starting at 0
     so they stay lined up, drums first. One undo removes them.

Stems are saved in `Stems/` beside the saved project (or
`~/.local/share/soniclab/stems` for an unsaved one), unless you choose
another folder with Change…. A song that was split before is found again
by its content, so choosing it shows its stems at once without Demucs.

## Models

| Model | Stems | Notes |
|---|---|---|
| `htdemucs_6s` (panel default) | vocals, drums, bass, guitar, piano, other | Demucs calls it experimental and says its piano "is not working great at the moment" |
| `htdemucs_ft` | drums, bass, other, vocals | best four-stem quality, four passes per song; what "Split Songs into Stems on Import" uses |
| `htdemucs` | drums, bass, other, vocals | one pass, fastest |

Demucs always writes every stem of the model; the tick boxes only choose
what is exported or added.

**Snare.** No Demucs model has a snare stem: the snare is inside *drums*,
and the panel never labels anything else as snare. Separate snare
extraction is Phase E of `docs/macos_plan.md` and is not built. The one
model found so far (LarsNet) has non-commercial weights.

## Devices

| Choice | What runs |
|---|---|
| Auto | Demucs's own default: an NVIDIA GPU if PyTorch finds one, otherwise the CPU. Never the Apple GPU |
| CPU | the CPU |
| NVIDIA GPU (CUDA) | listed only when PyTorch reports CUDA |
| Apple GPU (MPS, experimental) | listed only when PyTorch reports MPS. Not verified with Demucs. Unsupported operations fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`), and if the run still fails it is repeated on the CPU and the panel says so |

The device list comes from asking the Demucs environment once, on a
background thread, when the panel is first shown.

## Installing the separation engine

SonicLab never imports PyTorch. It starts Demucs as a separate process
from its own Python environment, which it looks for in this order:

1. the interpreter named by `SONICLAB_DEMUCS_PYTHON`
2. `.venv-demucs/bin/python` in the SonicLab folder
3. `~/.venvs/demucs/bin/python`

To create the third one (macOS or Linux; about 2 GB of downloads for
PyTorch on Linux, less on macOS):

```bash
python3 -m venv ~/.venvs/demucs
```

```bash
~/.venvs/demucs/bin/pip install demucs soundfile
```

```bash
~/.venvs/demucs/bin/python -m demucs --help
```

These commands have not been run from this guide yet. Notes from the
Demucs README: it needs Python 3.10 or newer, and on **Intel Macs** PyTorch
stopped at version 2.2, which needs Python 3.12 or older.

The panel shows what it found ("Demucs 4.x · PyTorch 2.x"), or "Demucs is
not installed" with the folder it looked in. An environment folder that
exists but has no Demucs in it gives the second message.

**Model weights** are not part of SonicLab or the plugins. Demucs
downloads a model the first time it is used (tens to a few hundred MB)
into PyTorch's cache (`~/.cache/torch/hub/checkpoints`); the panel says
whether the chosen model is already there. Demucs's code is MIT; its
README says the licence of the weights "is not clear". See
`docs/macos_plan.md` §5 before distributing a product built on them.

## How it works

* `soniclab/core/stems.py`: models, devices, `StemSplitter` (one Demucs
  process), `StemJobQueue` (one split at a time on a worker thread),
  `probe_engine`, `export_stems`. Demucs writes into a `.partial` folder
  and the stems move into place only after a successful run, so a
  cancelled or failed split never leaves stems that look finished.
* `soniclab/core/preview.py`: `PreviewPlayer`, a small output stream of
  its own for listening to a file. The project's `AudioEngine` is not
  involved or changed; starting project playback stops the preview.
* `soniclab/ui/stem_splitter.py`: the panel. It only polls: reading the
  song, probing the engine and splitting all happen off the UI thread.
* Folders: `<Stems>/<song>-<hash>/` holds the default model's stems (as
  before) and `<Stems>/<song>-<hash>/<model>/` the other models'.

## Not done or not verified

* A real Demucs run through the panel, including the six-stem model,
  piano quality, timing, memory on long songs, and MPS.
* A low-memory option. `StemSplitter.split(segment=...)` passes Demucs's
  `--segment`, but the panel has no control for it.
* An in-app installer for the engine; today it is the three commands above.
* Stems are not grouped in the project, and the project file does not
  record which model made them.
* Snare extraction, and macOS packaging.
