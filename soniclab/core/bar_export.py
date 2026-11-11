"""Write a range of bars to WAV, MP3 or OGG.

Bar edges are rounded to the nearest sample once, so the file is exactly
``end - start`` samples long and starts on the bar line. MP3 and OGG are
written through libsndfile, which records the encoder delay and padding, so
gapless decoders give back the same sample count. 16-bit WAV is dithered.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from soniclab.core.bar_map import bar_span


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    suffix: str
    format: str
    subtype: str
    compression_level: float | None = None
    bitrate_mode: str | None = None


FORMATS = (
    ExportFormat("wav24", "WAV · 24-bit", ".wav", "WAV", "PCM_24"),
    ExportFormat("wav16", "WAV · 16-bit (dithered)", ".wav", "WAV", "PCM_16"),
    ExportFormat("wav32", "WAV · 32-bit float", ".wav", "WAV", "FLOAT"),
    ExportFormat("mp3_320", "MP3 · 320 kbps", ".mp3", "MP3", "MPEG_LAYER_III", 0.0, "CONSTANT"),
    ExportFormat("mp3_v0", "MP3 · VBR V0", ".mp3", "MP3", "MPEG_LAYER_III", 0.0, "VARIABLE"),
    ExportFormat("ogg_q10", "OGG Vorbis · highest", ".ogg", "OGG", "VORBIS", 0.0),
    ExportFormat("ogg_q6", "OGG Vorbis · high", ".ogg", "OGG", "VORBIS", 0.4),
)
FORMATS_BY_KEY = {item.key: item for item in FORMATS}
# Largest 24-bit sample over full scale; 1.0 itself would wrap.
_PCM24_MAX = 1.0 - 2.0**-23


@dataclass(frozen=True)
class ExportResult:
    path: Path
    frames: int
    peak: float
    clipped: int


def bar_frames(lines, first: int, last: int, fallback: float, rate: int) -> tuple[int, int]:
    """Sample range from the start of bar ``first`` to the end of bar ``last``.

    Bars are zero-based and inclusive.
    """
    if last < first:
        first, last = last, first
    start, _end = bar_span(lines, max(0, first), fallback)
    _start, end = bar_span(lines, max(0, last), fallback)
    return int(round(start * rate)), int(round(end * rate))


def default_name(project: str, first: int, last: int, track: str | None = None) -> str:
    bars = f"bar {first + 1}" if first == last else f"bars {first + 1}-{last + 1}"
    parts = [project or "Untitled"] + ([track] if track else []) + [bars]
    return re.sub(r'[\\/:*?"<>|]+', "_", " ".join(parts)).strip()


def with_suffix(path, fmt: ExportFormat) -> Path:
    path = Path(path)
    if path.suffix.lower() != fmt.suffix:
        path = path.with_name(path.name + fmt.suffix) if path.suffix.lower() not in (".wav", ".mp3", ".ogg") else path.with_suffix(fmt.suffix)
    return path


def _dither16(audio: np.ndarray) -> np.ndarray:
    # TPDF dither of one LSB, seeded so the same bars export the same bytes.
    rng = np.random.default_rng(0x50C1C)
    noise = rng.random(audio.shape) - rng.random(audio.shape)
    scaled = np.round(audio.astype(np.float64) * 32767.0 + noise)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def write_audio(path, audio, rate: int, fmt: ExportFormat) -> ExportResult:
    """Write ``audio`` (frames x 2, float) to ``path`` in ``fmt``, atomically."""
    path = with_suffix(path, fmt)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("Nothing to export")
    peak = float(np.max(np.abs(audio)))
    clipped = int(np.count_nonzero(np.abs(audio) >= 1.0))
    if fmt.subtype == "PCM_16":
        data = _dither16(audio)
    elif fmt.subtype == "PCM_24":
        data = np.clip(audio, -1.0, _PCM24_MAX)
    else:
        data = np.clip(audio, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".export-", suffix=fmt.suffix, dir=path.parent)
    os.close(handle)
    try:
        options = {}
        if fmt.compression_level is not None:
            options["compression_level"] = fmt.compression_level
        if fmt.bitrate_mode is not None:
            options["bitrate_mode"] = fmt.bitrate_mode
        sf.write(temporary, data, int(rate), subtype=fmt.subtype, format=fmt.format, **options)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return ExportResult(path, int(audio.shape[0]), peak, clipped)
