#!/usr/bin/env python3
"""Prove this Python can track a click with madmom-infer.

The pretrained weights are downloaded into the user cache on first use.
They are CC BY-NC-SA 4.0 and are not bundled with SonicLab.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path


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
    position = int(rate * 0.05)
    while position < count - 200:
        for offset in range(200):
            encoded = int(26000).to_bytes(2, "little", signed=True)
            index = (position + offset) * 2
            frames[index : index + 2] = encoded
        position += interval
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def main() -> int:
    try:
        import madmom_infer
        from madmom_infer.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )
    except Exception as exc:
        return fail(f"madmom-infer could not be imported: {exc}")
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/soniclab-infer-click.wav")
    click_wav(wav)
    try:
        activations = RNNDownBeatProcessor()(str(wav))
        tracked = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(activations)
    except Exception as exc:
        text = str(exc)
        if "sha256" in text or "download" in text.lower() or "urlopen" in text.lower():
            return fail(f"Beat model download failed: {exc}")
        return fail(f"Beat tracking failed: {exc}")
    if len(tracked) < 8:
        return fail("Beat tracking ran, but it did not find the click track.")
    intervals = [float(tracked[index][0] - tracked[index - 1][0]) for index in range(1, len(tracked))]
    intervals.sort()
    tempo = 60.0 / intervals[len(intervals) // 2]
    if not 110.0 <= tempo <= 130.0:
        return fail(f"The click track came back at {tempo:.1f} BPM instead of about 120.")
    version = getattr(madmom_infer, "__version__", "")
    print(f"SONICLAB_INFER_OK madmom-infer={version} beats={len(tracked)} tempo={tempo:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
