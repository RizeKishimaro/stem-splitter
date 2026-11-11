"""Pick the drum track that every other track takes its downbeats from.

The file name is checked first (``drums.wav``, ``Kick``, ``perc loop``). When
no name matches, the track whose sound is most percussive wins. Percussive
share comes from median-filter harmonic/percussive separation: drums are
broadband and short, so they survive a median across frequency, while held
notes survive a median across time.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

from soniclab.core.waveform import mono

ANALYSIS_RATE = 11025
EXCERPTS = 6
EXCERPT_SECONDS = 8.0
WINDOW = 1024
HOP = 256
KERNEL = 17
# Frames quieter than -40 dB under the loudest frame are ignored.
LOUD_FRACTION = 0.01
ACTIVE_RMS = 0.01
# A track sounding in at least this share of the file is not penalized.
FULL_ACTIVITY = 0.4
# Measured on Demucs stems: drums 0.50-0.75, bass under 0.3, vocals and
# other parts mostly under 0.48. The most drum-like track above this wins.
MIN_PERCUSSIVE_SHARE = 0.45

_PREFIXES = ("drum", "perc")
_WORDS = {
    "kit",
    "kick",
    "kicks",
    "snare",
    "snares",
    "hat",
    "hats",
    "hihat",
    "hihats",
    "cymbal",
    "cymbals",
    "tom",
    "toms",
    "overhead",
    "overheads",
    "breakbeat",
}


def name_says_drums(name: str) -> bool:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name or ""))
    for word in re.split(r"[^a-z0-9]+", spaced.lower()):
        if not word:
            continue
        if word in _WORDS or word.startswith(_PREFIXES):
            return True
    return False


def percussive_share(audio, sample_rate: int) -> float:
    """How drum-like a track sounds, 0..1. Silence scores 0.

    The mean of two measures over the loud frames of a few excerpts spread
    across the file: the magnitude share of the percussive part, and the
    spectral flatness (noise-like, as cymbals and snares are). Bass has a
    strong percussive attack but almost no flatness, so it stays low.
    """
    samples = mono(audio)
    rate = int(sample_rate)
    if rate <= 0 or samples.size == 0:
        return 0.0
    step = max(1, rate // ANALYSIS_RATE)
    samples = samples[::step]
    excerpt = int(EXCERPT_SECONDS * rate / step)
    if samples.size < WINDOW * 4:
        return 0.0
    if samples.size > excerpt * EXCERPTS:
        starts = np.linspace(0, samples.size - excerpt, EXCERPTS).astype(int)
        pieces = [samples[start : start + excerpt] for start in starts]
    else:
        pieces = [samples]
    magnitudes = []
    masks = []
    for piece in pieces:
        _freqs, _times, spectrum = stft(
            piece, fs=rate / step, nperseg=WINDOW, noverlap=WINDOW - HOP, boundary=None
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        harmonic = median_filter(magnitude, size=(1, KERNEL), mode="nearest")
        percussive = median_filter(magnitude, size=(KERNEL, 1), mode="nearest")
        masks.append(percussive**2 / (harmonic**2 + percussive**2 + 1e-12))
        magnitudes.append(magnitude)
    magnitude = np.concatenate(magnitudes, axis=1)
    mask = np.concatenate(masks, axis=1)
    frames = magnitude.sum(axis=0)
    peak = float(frames.max()) if frames.size else 0.0
    if peak <= 1e-9:
        return 0.0
    loud = frames >= peak * LOUD_FRACTION
    if int(loud.sum()) < 8:
        return 0.0
    magnitude = magnitude[:, loud]
    share = float((mask[:, loud] * magnitude).sum() / magnitude.sum())
    geometric = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
    flatness = float(np.median(geometric / (magnitude.mean(axis=0) + 1e-10)))
    return 0.5 * (share + flatness) * min(1.0, activity(samples, rate / step) / FULL_ACTIVITY)


def activity(samples, sample_rate: float) -> float:
    """Fraction of 100 ms blocks louder than -40 dBFS RMS.

    A mostly silent stem has a noise floor that looks flat and percussive.
    Drums usually play through most of a song.
    """
    block = max(1, int(sample_rate * 0.1))
    count = samples.size // block
    if count == 0:
        return 0.0
    blocks = samples[: count * block].reshape(count, block)
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    return float(np.mean(rms > ACTIVE_RMS))


def pick_drum_track(tracks, sample_rate: int, score=percussive_share) -> int | None:
    """Index of the drum track, or None when nothing sounds like drums.

    A track already marked ``drum_reference`` keeps the role. Scores are
    cached on the track as ``_drum_score``.
    """
    for index, track in enumerate(tracks):
        if track.get("drum_reference"):
            return index
    for index, track in enumerate(tracks):
        if _has_audio(track) and (
            name_says_drums(track.get("name", ""))
            or name_says_drums(_stem(track.get("path")))
        ):
            return index
    best = None
    best_score = MIN_PERCUSSIVE_SHARE
    for index, track in enumerate(tracks):
        if not _has_audio(track):
            continue
        value = track.get("_drum_score")
        if value is None:
            value = float(score(track["_audio"], sample_rate))
            track["_drum_score"] = value
        if value >= best_score:
            best, best_score = index, value
    return best


def _has_audio(track) -> bool:
    audio = track.get("_audio")
    return audio is not None and len(audio) > 1 and not track.get("empty")


def _stem(path) -> str:
    if not path:
        return ""
    return re.sub(r"\.[^./]+$", "", str(path).replace("\\", "/").rsplit("/", 1)[-1])
