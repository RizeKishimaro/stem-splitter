"""Read a song before Demucs or the stem panel touches it.

SoundFile handles WAV, FLAC, AIFF, OGG, and MP3 when libsndfile was built
with those codecs. Anything else is converted with FFmpeg to a float WAV.
TorchAudio is not used. TorchAudio 2.9 and later ask TorchCodec for a
backend, and a missing backend is the "format None" error.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioLoadError(Exception):
    """The song cannot be decoded. The message is safe to show in the window."""


@dataclass
class PreparedAudio:
    """A file Demucs can read, plus how to remove a temporary conversion."""

    path: Path
    sample_rate: int
    channels: int
    frames: int
    temporary: Path | None = None

    def cleanup(self) -> None:
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)


def read_audio(path) -> tuple[np.ndarray, int]:
    """Return float32 audio shaped ``(frames, channels)`` and its sample rate."""
    prepared = prepare_demucs_input(path, directory=None)
    try:
        audio, rate = sf.read(str(prepared.path), dtype="float32", always_2d=True)
    finally:
        prepared.cleanup()
    _check_array(audio, rate, Path(path))
    return audio, int(rate)


def prepare_demucs_input(path, directory: Path | None = None) -> PreparedAudio:
    """Return a path Demucs can open.

    SoundFile-readable files are passed through. Other files are converted
    to a float WAV. ``directory`` receives that WAV when it is set; otherwise
    a temporary file is used.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise AudioLoadError(f"Audio file not found: {source}")
    info = _soundfile_info(source)
    if info is not None:
        return PreparedAudio(source.resolve(), info.samplerate, info.channels, info.frames)
    dest = _destination(source, directory)
    _ffmpeg_to_wav(source, dest)
    converted = _soundfile_info(dest)
    if converted is None:
        dest.unlink(missing_ok=True)
        raise AudioLoadError(_unreadable(source))
    return PreparedAudio(
        dest.resolve(),
        converted.samplerate,
        converted.channels,
        converted.frames,
        temporary=dest,
    )


def explain_separation_error(detail: str) -> str | None:
    """Turn a TorchAudio or Demucs decoder traceback into one sentence."""
    text = detail.lower()
    if "appropriate backend" in text or "torchcodec" in text or "format none" in text:
        return (
            "The audio decoder could not open this file. "
            "Convert it to WAV and try again."
        )
    if "trying to load using" in text or "ffmpeg could not read" in text:
        return "This audio file could not be decoded. Convert it to WAV and try again."
    if "ffmpeg is not installed" in text:
        return "FFmpeg is not installed, so this format could not be read. Convert it to WAV and try again."
    return None


def _soundfile_info(path: Path):
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    if info.frames <= 0 or info.channels <= 0 or info.samplerate <= 0:
        raise AudioLoadError(f"The audio file is empty: {path.name}")
    try:
        chunk, _rate = sf.read(str(path), dtype="float32", always_2d=True, frames=1)
    except Exception as exc:
        raise AudioLoadError(_unreadable(path)) from exc
    if chunk.size == 0:
        raise AudioLoadError(f"The audio file is empty: {path.name}")
    return info


def _check_array(audio: np.ndarray, rate: int, source: Path) -> None:
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0 or rate <= 0:
        raise AudioLoadError(f"The audio file is empty: {source.name}")
    if audio.dtype != np.float32:
        raise AudioLoadError(f"Could not read {source.name} as floating-point audio.")


def _destination(source: Path, directory: Path | None) -> Path:
    if directory is None:
        handle, name = tempfile.mkstemp(prefix="soniclab-audio-", suffix=".wav")
        import os
        os.close(handle)
        return Path(name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / ".decoded.wav"


def _ffmpeg_to_wav(source: Path, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError(
            f"Could not read {source.name}. SoundFile does not support this format, "
            "and FFmpeg is not installed. Convert it to WAV and try again."
        )
    done = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-f", "wav", "-acodec", "pcm_f32le", str(dest)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or not dest.is_file() or dest.stat().st_size < 64:
        dest.unlink(missing_ok=True)
        raise AudioLoadError(_unreadable(source))


def _unreadable(source: Path) -> str:
    return (
        f"Could not read {source.name}. The file is damaged or its format is unsupported. "
        "Convert it to WAV and try again."
    )
