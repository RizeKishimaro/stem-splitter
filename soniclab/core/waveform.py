"""Min/max waveform envelopes computed off the audio thread.

The timeline draws one filled shape from these arrays. It does not turn
each bucket into its own graphics item.
"""

import weakref

import numpy as np

# One peak pair per 32 samples is finer than a pixel at the closest zoom.
SAMPLES_PER_BUCKET = 32

_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
# The same buffer again (a canvas rebuild, the other canvas): found by
# identity, without mixing it to mono just to fingerprint it.
_BY_BUFFER: dict[int, tuple] = {}


def mono(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def peak_envelope(audio, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bucket minimum and maximum.

    When the audio is shorter than `buckets`, each sample is its own bucket
    so a closer zoom still shows the samples themselves.
    """
    samples = mono(audio)
    buckets = max(1, int(buckets))
    if samples.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    if samples.size <= buckets:
        values = samples.astype(np.float32, copy=True)
        return values, values.copy()
    starts = (np.arange(buckets, dtype=np.int64) * samples.size) // buckets
    lows = np.minimum.reduceat(samples, starts).astype(np.float32, copy=False)
    highs = np.maximum.reduceat(samples, starts).astype(np.float32, copy=False)
    return lows, highs


def pixel_columns(lows, highs, width: int) -> tuple[np.ndarray, np.ndarray]:
    """One upper and lower peak per clip pixel, ready to paint a narrow strip."""
    width = max(1, int(width))
    if lows.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    upper = np.maximum(highs, 0.0)
    lower = np.minimum(lows, 0.0)
    flat = (highs - lows) < 1e-4
    peak = np.maximum(np.abs(lows), np.abs(highs))
    upper = np.where(flat, peak, upper).astype(np.float32, copy=False)
    lower = np.where(flat, -peak, lower).astype(np.float32, copy=False)
    count = int(upper.size)
    if count == width:
        return upper, lower
    if count == 1:
        return (
            np.full(width, upper[0], dtype=np.float32),
            np.full(width, lower[0], dtype=np.float32),
        )
    index = np.linspace(0, count - 1, width).astype(np.int64)
    return upper[index], lower[index]


def display_envelope(lows, highs, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a cached envelope to the clip's current pixel width."""
    pixels = max(1, int(pixels))
    if lows.size == 0:
        return lows, highs
    if lows.size <= pixels:
        return lows, highs
    starts = (np.arange(pixels, dtype=np.int64) * lows.size) // pixels
    return (
        np.minimum.reduceat(lows, starts).astype(np.float32, copy=False),
        np.maximum.reduceat(highs, starts).astype(np.float32, copy=False),
    )


def _fingerprint(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    step = max(1, samples.size // 16)
    return np.ascontiguousarray(samples[::step][:16]).tobytes()


def cached_envelope(audio) -> tuple[np.ndarray, np.ndarray]:
    """High-resolution peaks for one buffer. Repeated calls reuse the cache."""
    key = _buffer_key(audio)
    if key is not None:
        known = _BY_BUFFER.get(key[0])
        if known is not None and known[0]() is key[1]:
            return known[1]
    found = _envelope(audio)
    if key is not None:
        if len(_BY_BUFFER) > 64:
            _BY_BUFFER.clear()
        _BY_BUFFER[key[0]] = (weakref.ref(key[1]), found)
    return found


def _buffer_key(audio):
    """``(key, owner)`` for a numpy buffer or a slice of one: the owning
    array, where the slice starts in it and its shape. None otherwise."""
    if not isinstance(audio, np.ndarray):
        return None
    owner = audio
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    start = audio.__array_interface__["data"][0] - owner.__array_interface__["data"][0]
    return (id(owner), start, audio.shape, audio.strides, audio.dtype.str), owner


def _envelope(audio) -> tuple[np.ndarray, np.ndarray]:
    samples = mono(audio)
    key = (int(samples.size), _fingerprint(samples))
    found = _CACHE.get(key)
    if found is not None:
        return found
    buckets = max(1, int(np.ceil(samples.size / SAMPLES_PER_BUCKET))) if samples.size else 1
    found = peak_envelope(samples, buckets)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = found
    return found
