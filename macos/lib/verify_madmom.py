#!/usr/bin/env python3
"""Prove this Python can load Madmom's downbeat models and track a click.

Run it with the Madmom environment, not with SonicLab's main environment.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

MIN_MODEL_BYTES = 100_000


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def click_wav(path: Path) -> None:
    rate = 44100
    seconds = 16
    bpm = 120
    count = rate * seconds
    frames = bytearray(count * 2)
    interval = int(rate * 60 / bpm)
    start = int(rate * 0.05)
    position = start
    while position < count - 200:
        for offset in range(200):
            sample = 26000
            encoded = int(sample).to_bytes(2, "little", signed=True)
            index = (position + offset) * 2
            frames[index : index + 2] = encoded
        position += interval
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        return fail(f"Madmom needs Python 3.11. This one is {sys.version.split()[0]}.")
    try:
        import madmom
        import numpy
        import scipy
        from importlib.metadata import version
    except Exception as exc:
        return fail(f"Madmom could not be imported: {exc}")
    expected = {
        "madmom": "0.16.1",
        "numpy": "1.23.5",
        "scipy": "1.10.1",
    }
    found = {
        "madmom": str(getattr(madmom, "__version__", "")),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    for name, want in expected.items():
        if found[name] != want:
            return fail(f"{name} is {found[name]}, not {want}.")
    try:
        setuptools_version = version("setuptools")
    except Exception as exc:
        return fail(f"setuptools could not be checked: {exc}")
    major = int(setuptools_version.split(".", 1)[0])
    if major >= 81:
        return fail(f"setuptools {setuptools_version} removes a library Madmom still imports.")
    if not hasattr(numpy, "float") or not hasattr(numpy, "int"):
        return fail("This NumPy is too new for Madmom.")

    from madmom.models import DOWNBEATS_BLSTM

    if len(DOWNBEATS_BLSTM) != 8:
        return fail(f"Expected 8 downbeat models, found {len(DOWNBEATS_BLSTM)}.")
    for model in DOWNBEATS_BLSTM:
        file = Path(model)
        if not file.is_file() or file.stat().st_size < MIN_MODEL_BYTES:
            return fail(f"Downbeat model is missing or too small: {file.name}")

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/soniclab-madmom-click.wav")
    click_wav(wav)
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    activations = RNNDownBeatProcessor()(str(wav))
    if getattr(activations, "ndim", 0) != 2 or activations.shape[1] != 2 or activations.shape[0] < 100:
        return fail("The downbeat model did not return beat and downbeat activations.")
    tracked = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(activations)
    if len(tracked) < 8:
        return fail("Beat tracking ran, but it did not find the click track.")
    intervals = [float(tracked[index][0] - tracked[index - 1][0]) for index in range(1, len(tracked))]
    intervals.sort()
    median = intervals[len(intervals) // 2]
    tempo = 60.0 / median
    if not 110.0 <= tempo <= 130.0:
        return fail(f"The click track came back at {tempo:.1f} BPM instead of about 120.")
    print(
        f"SONICLAB_MADMOM_OK madmom={found['madmom']} numpy={found['numpy']} "
        f"scipy={found['scipy']} models=8 beats={len(tracked)} tempo={tempo:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
