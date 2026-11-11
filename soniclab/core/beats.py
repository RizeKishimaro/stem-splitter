"""Beat times from aubio. This never runs inside the audio callback."""

from dataclasses import dataclass

import numpy as np

from soniclab.core.waveform import mono

try:
    import aubio
except ImportError:  # pragma: no cover - the package is a project dependency
    aubio = None


@dataclass(frozen=True)
class BeatAnalysis:
    beats: tuple[float, ...]
    tempo: float


_CACHE: dict[tuple, BeatAnalysis] = {}
_EMPTY = BeatAnalysis((), 0.0)


def _fingerprint(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    step = max(1, samples.size // 24)
    return np.ascontiguousarray(samples[::step][:24]).tobytes()


def analyze_beats(audio, sample_rate: float) -> BeatAnalysis:
    """Return detected beat times in seconds and the tracked tempo.

    Positions come from aubio's beat tracker. A file with no detected
    pulse returns an empty list rather than a generated grid.
    """
    samples = mono(audio)
    rate = int(sample_rate)
    if aubio is None or rate <= 0 or samples.size < 1024:
        return _EMPTY
    key = (rate, int(samples.size), _fingerprint(samples))
    found = _CACHE.get(key)
    if found is not None:
        return found

    window = 1024
    hop = 512
    tracker = aubio.tempo("default", window, hop, rate)
    beats: list[float] = []
    limit = samples.size - hop + 1
    for start in range(0, limit, hop):
        detected = tracker(samples[start : start + hop])
        if float(detected[0]) <= 0.0:
            continue
        beat = float(tracker.get_last_s())
        if not np.isfinite(beat) or beat < 0.0:
            continue
        if beats and abs(beat - beats[-1]) < 1e-3:
            continue
        beats.append(beat)
    tempo = float(tracker.get_bpm()) if beats else 0.0
    if not np.isfinite(tempo) or tempo <= 0.0:
        tempo = 0.0
    result = BeatAnalysis(tuple(beats), tempo)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = result
    return result
