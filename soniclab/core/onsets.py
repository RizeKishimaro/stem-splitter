"""Aubio complex onsets. This never runs inside the audio callback.

Timestamps are seconds from the start of the source audio. They are not
beats and they are not Madmom downbeats. Results are cached by file SHA-256.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from soniclab.core.beat_analysis import file_sha256
from soniclab.core.waveform import mono

try:
    import aubio
except ImportError:  # pragma: no cover - aubio is a project dependency
    aubio = None


METHOD = "complex"
MIN_INTERVAL_SECONDS = 0.05
CACHE_SCHEMA_VERSION = 1
WINDOW = 1024
HOP = 512

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
STATUS_UNAVAILABLE = "unavailable"
STATUS_CANCELLED = "cancelled"


@dataclass(frozen=True)
class OnsetResult:
    status: str
    onsets: tuple[float, ...] = ()
    message: str = ""
    cached: bool = False


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "soniclab" / "onsets"


def detect_onsets(audio, sample_rate: float, cancel_event=None) -> tuple[float, ...]:
    """Return onset times in seconds. A quiet buffer returns an empty tuple."""
    samples = np.ascontiguousarray(mono(audio), dtype=np.float32)
    rate = int(sample_rate)
    if aubio is None or rate <= 0 or samples.size < HOP:
        return ()
    detector = aubio.onset(METHOD, WINDOW, HOP, rate)
    detector.set_minioi_ms(MIN_INTERVAL_SECONDS * 1000.0)
    detector.set_silence(-70.0)
    found: list[float] = []
    limit = samples.size - HOP + 1
    for start in range(0, limit, HOP):
        if cancel_event is not None and cancel_event.is_set():
            return ()
        detected = detector(samples[start : start + HOP])
        if float(detected[0]) <= 0.0:
            continue
        onset = float(detector.get_last_s())
        if not np.isfinite(onset) or onset <= 0.0:
            continue
        if found and onset - found[-1] < MIN_INTERVAL_SECONDS - 1e-4:
            continue
        found.append(onset)
    return tuple(found)


def analyze_path(path, cache_dir: Path | None = None, cancel_event=None) -> OnsetResult:
    """Read a source file in place and return cached or newly detected onsets."""
    source = Path(path)
    if not source.is_file():
        return OnsetResult(STATUS_ERROR, message="Source file is missing")
    if aubio is None:
        return OnsetResult(STATUS_UNAVAILABLE, message="Aubio is not installed")
    try:
        digest = file_sha256(source)
    except OSError as exc:
        return OnsetResult(STATUS_ERROR, message=str(exc))
    cache_path = _cache_path(digest, cache_dir)
    cached = _read_cache(cache_path, digest)
    if cached is not None:
        status = STATUS_OK if cached else STATUS_EMPTY
        return OnsetResult(status, cached, cached=True)
    if cancel_event is not None and cancel_event.is_set():
        return OnsetResult(STATUS_CANCELLED, message="Onset analysis cancelled")
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        return OnsetResult(STATUS_ERROR, message=str(exc))
    onsets = detect_onsets(audio, sample_rate, cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        return OnsetResult(STATUS_CANCELLED, message="Onset analysis cancelled")
    _write_cache(cache_path, digest, onsets)
    status = STATUS_OK if onsets else STATUS_EMPTY
    return OnsetResult(status, onsets)


def _cache_path(digest: str, cache_dir: Path | None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / f"{digest}.json"


def _read_cache(path: Path, digest: str) -> tuple[float, ...] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        document.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or document.get("detector") != "aubio"
        or document.get("method") != METHOD
        or document.get("min_interval_seconds") != MIN_INTERVAL_SECONDS
        or document.get("content_sha256") != digest
    ):
        return None
    onsets = document.get("onsets")
    if not isinstance(onsets, list):
        return None
    times: list[float] = []
    for value in onsets:
        try:
            onset = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(onset) or onset <= 0.0:
            return None
        if times and onset <= times[-1]:
            return None
        times.append(onset)
    return tuple(times)


def _write_cache(path: Path, digest: str, onsets: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "detector": "aubio",
        "method": METHOD,
        "min_interval_seconds": MIN_INTERVAL_SECONDS,
        "content_sha256": digest,
        "onsets": list(onsets),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    os.replace(temporary, path)
