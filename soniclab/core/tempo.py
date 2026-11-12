"""Change a recording's tempo without changing its pitch.

WSOLA (waveform-similarity overlap-add, Verhelst and Roelands, 1993): the
output is built from short overlapping pieces of the original waveform.
Each piece is taken from about where the new tempo says it should come
from, shifted by up to ±12 ms to wherever it best continues the piece
before it. Because the pieces are the original samples, drum hits keep
their attack and level; a phase vocoder smears them. Every channel uses
the same pieces, so the stereo image stays intact.

``factor`` is the time scale: 2.0 makes the audio twice as long (half the
tempo), 0.8 makes it 20 % shorter (a 25 % faster tempo). The result is
``round(len(audio) * factor)`` frames long, so a stretched clip is exactly
as long as its new duration says. Best between about 0.5 and 2.

This is offline work, about a second per minute of stereo audio: never
call it on the audio callback or the UI thread.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

FRAME = 1024  # piece length (23 ms at 44.1 kHz)
HOP_OUT = FRAME // 2
SEARCH = 512  # how far a piece may move to match (±12 ms)


def tempo_factor(old_bpm: float, new_bpm: float) -> float:
    """Time scale that turns ``old_bpm`` into ``new_bpm``."""
    if old_bpm <= 0 or new_bpm <= 0:
        raise ValueError("tempos must be positive")
    return float(old_bpm) / float(new_bpm)


def stretch(audio, factor: float, progress=None) -> np.ndarray:
    """``audio`` (frames, channels) or (frames,) stretched in time by ``factor``.

    ``progress(fraction)`` is called now and then from this thread.
    """
    samples = np.asarray(audio, dtype=np.float32)
    mono = samples.ndim == 1
    if mono:
        samples = samples[:, None]
    frames, channels = samples.shape
    target = int(round(frames * float(factor)))
    if frames == 0 or target <= 0:
        shape = (max(0, target),) if mono else (max(0, target), channels)
        return np.zeros(shape, dtype=np.float32)
    if abs(factor - 1.0) < 1e-6:
        return samples[:, 0].copy() if mono else samples.copy()

    hop_in = HOP_OUT / float(factor)
    window = np.hanning(FRAME).astype(np.float32)
    pad = SEARCH + FRAME
    padded = np.concatenate(
        [np.zeros((pad, channels), np.float32), samples, np.zeros((pad + FRAME, channels), np.float32)]
    )
    guide = padded.mean(axis=1)  # pieces are chosen on the mono mix
    steps = int(np.ceil(target / HOP_OUT)) + 2
    out = np.zeros((steps * HOP_OUT + FRAME, channels), dtype=np.float64)
    norm = np.zeros(steps * HOP_OUT + FRAME, dtype=np.float64)
    taken = None  # where the previous piece started in ``padded``
    for step in range(steps):
        # The piece centred on the input time this output time stands for.
        nominal = pad + int(round(step * hop_in)) - FRAME // 2
        if nominal + FRAME + SEARCH > len(padded):
            break
        if taken is None:
            start = nominal
        else:
            # The previous piece's natural continuation, half a frame on.
            follow = guide[taken + HOP_OUT : taken + HOP_OUT + FRAME]
            low = max(0, nominal - SEARCH)
            region = guide[low : nominal + SEARCH + FRAME]
            if np.any(follow) and np.any(region):
                score = fftconvolve(region, follow[::-1], mode="valid")
                start = low + int(np.argmax(score))
            else:
                start = nominal
        piece = padded[start : start + FRAME] * window[:, None]
        at = step * HOP_OUT
        out[at : at + FRAME] += piece
        norm[at : at + FRAME] += window
        taken = start
        if progress is not None and step % 512 == 0:
            try:
                progress(step / steps)
            except Exception:
                pass
    norm[norm < 1e-3] = 1.0
    result = out / norm[:, None]
    # Output frame step*HOP_OUT holds the piece centred on input step*hop_in,
    # so output sample FRAME/2 is input sample 0.
    begin = FRAME // 2
    result = result[begin : begin + target]
    if len(result) < target:
        result = np.concatenate([result, np.zeros((target - len(result), channels))])
    result = result.astype(np.float32)
    return result[:, 0] if mono else result
