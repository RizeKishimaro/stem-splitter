"""Decode audio files off the GUI thread.

``submit`` starts decoding a batch of files on worker threads and returns at
once. ``poll`` (called from the GUI thread, e.g. on the transport timer)
hands each finished batch to its callback, in the order batches were
submitted, so tracks are added in one step with one canvas refresh. libsndfile
and the resampler release the GIL, so the window keeps painting meanwhile.
"""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.beats import analyze_beats
from soniclab.core.waveform import cached_envelope

# Two workers load almost as fast as four, and leave the GUI thread more
# turns at the interpreter, so the window stays smoother while they run.
WORKERS = 2


def decode(path: str):
    """Float32 frames x channels at the engine's rate."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioEngine.resample(audio, sample_rate)


@dataclass(frozen=True)
class Loaded:
    """A decoded file with what drawing its track needs, measured on the
    worker too: beat positions, tempo and file peak. Its waveform envelope
    is in the waveform cache by then."""

    audio: np.ndarray
    beats: tuple
    tempo: float
    peak_db: str

    def track_fields(self) -> dict:
        return {"_audio": self.audio, "_beats": self.beats, "_tempo": self.tempo, "_peak_db": self.peak_db}


def peak_db(audio) -> str:
    if audio is None or len(audio) == 0:
        return "—"
    peak = float(np.max(np.abs(np.asarray(audio))))
    if peak <= 1e-8:
        return "−∞"
    return f"{20.0 * math.log10(peak):+.1f}"


def load(path: str) -> Loaded:
    audio = decode(path)
    cached_envelope(audio)
    analysis = analyze_beats(audio, OUTPUT_RATE)
    return Loaded(audio, tuple(analysis.beats), float(analysis.tempo), peak_db(audio))


class AudioLoader:
    def __init__(self, workers: int = WORKERS):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="soniclab-decode")
        self._batches: list[tuple[list[Future], object]] = []

    def submit(self, paths, done) -> None:
        """Load ``paths``; later ``done(results)`` runs on the GUI thread.

        ``results`` maps each path to its ``Loaded``, or to the exception
        that loading it raised.
        """
        paths = list(paths)
        futures = [self._pool.submit(load, path) for path in paths]
        for path, future in zip(paths, futures):
            future.path = path
        self._batches.append((futures, done))

    def pending(self) -> bool:
        return bool(self._batches)

    def poll(self) -> None:
        while self._batches and all(future.done() for future in self._batches[0][0]):
            futures, done = self._batches.pop(0)
            results = {}
            for future in futures:
                error = future.exception()
                results[future.path] = error if error is not None else future.result()
            done(results)

    def wait(self) -> None:
        """Finish every batch now (tests, and closing the window)."""
        while self._batches:
            wait(self._batches[0][0])
            self.poll()

    def shutdown(self) -> None:
        self._batches.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
