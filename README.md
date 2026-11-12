# SonicLab

A lightweight Python/PySide6 multitrack audio workstation foundation.

## Features in this first milestone

- Import WAV, FLAC, OGG and MP3 (MP3 support depends on the installed libsndfile build).
- Display audio clips and waveform previews on separate tracks.
- Move clips horizontally by dragging.
- Play multiple clips in sync through one audio output stream.
- Transport controls and a moving playhead.
- Save/open a simple JSON project.

This is an early foundation, not a finished DAW. Clip trimming, effects, automation,
advanced time-stretching and robust device selection are not implemented yet.

## Arch Linux setup (Fish shell)

```fish
sudo pacman -S --needed python python-pip python-virtualenv libsndfile portaudio
mkdir -p ~/Projects/soniclab
cd ~/Projects/soniclab
unzip ~/Downloads/soniclab-starter.zip
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -e .
python -m soniclab.main
```

If MP3 files fail to load, install a libsndfile build with MPEG support or convert
them to WAV/FLAC with FFmpeg.

Beat and downbeat analysis is optional. It uses madmom-infer, not the old Madmom package:

```fish
python -m venv .venv-madmom-infer
.venv-madmom-infer/bin/python -m pip install "madmom-infer @ git+https://github.com/openmirlab/madmom-infer.git@bffead9be61857fd44b2e0d3a2510d9d94c8f2d2"
```

SonicLab finds that interpreter on its own. The first analysis downloads the pretrained weights into `~/.cache/madmom_infer`. Those weights are CC BY-NC-SA 4.0. They are not commercially licensed, and downloading them does not change that. Files that are not 44100 Hz WAV need ffmpeg on `PATH`. `SONICLAB_MADMOM_LEGACY=1` keeps using `.venv-madmom` and `tools/madmom_worker.py`.

## Audio engine notes

The application uses one PortAudio output stream. All clips are mixed in the callback
at the same sample position, which keeps tracks synchronized. The initial engine uses
a fixed 44.1 kHz output rate and resamples imported files as needed.

## Project format

Projects are JSON files containing audio file paths, track names, clip start positions,
and basic project settings. Audio files are referenced, not copied into the project.
Keep source files in place or use the future project-collect feature.
