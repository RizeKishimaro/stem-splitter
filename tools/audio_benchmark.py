"""Measure SonicLab's real playback on this machine: audio and GUI together.

Opens the real window on the real audio output (no offscreen platform, no
fake stream), loads synthetic stems, plays them with the canvas following
the playhead, and prints what the engine and the GUI measured. Run it on the
Mac whose playback glitches; numbers from Linux say nothing about macOS.

    .venv/bin/python tools/audio_benchmark.py              # 6 stems, 20 s
    .venv/bin/python tools/audio_benchmark.py --stems 12 --seconds 30
    SONICLAB_AUDIO_LATENCY=low .venv/bin/python tools/audio_benchmark.py

It plays the song transport for the first half and block voices (one per
stem, on one song clock) for the second half. Output is quiet sine tones.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from soniclab.core.audio_engine import OUTPUT_RATE  # noqa: E402
from soniclab.ui.main_window import MainWindow  # noqa: E402
from soniclab.ui.timeline import FOLLOW_CENTER  # noqa: E402

NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano")


def make_stems(folder: Path, count: int, seconds: float) -> dict:
    frames = np.arange(int(seconds * OUTPUT_RATE)) / OUTPUT_RATE
    stems = {}
    for k in range(count):
        tone = (0.02 * np.sin(2 * np.pi * (220.0 + 55.0 * k) * frames)).astype(np.float32)
        name = NAMES[k] if k < len(NAMES) else f"stem{k}"
        path = folder / f"bench-{name}.wav"
        sf.write(path, np.column_stack((tone, tone)), OUTPUT_RATE, subtype="FLOAT")
        stems[name] = str(path)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stems", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    folder = Path(tempfile.mkdtemp(prefix="soniclab-bench-"))
    window = MainWindow()
    window.resize(1600, 1000)
    window.show()
    stems = make_stems(folder, args.stems, max(args.seconds, 8.0) + 2.0)
    added = window._add_stem_tracks("Bench", stems, tuple(stems))
    window._reload_engine()
    window._refresh_timeline()
    window._set_follow_mode(FOLLOW_CENTER)
    app.processEvents()

    half = args.seconds / 2.0
    began = time.monotonic()
    phase = {"name": "transport"}
    window._play()

    def switch():
        # Second half: every stem as a block voice on the song's clock.
        phase["name"] = "blocks"
        for index, _track in enumerate(added):
            window._toggle_block(index, index % 4)

    def finish():
        diagnostics = window.engine.diagnostics()
        print(f"\nSonicLab audio benchmark — {platform.system()} {platform.machine()} {platform.mac_ver()[0]}".rstrip())
        print(f"Qt platform: {app.platformName()} · {args.stems} stems · {args.seconds:.0f} s "
              f"(transport {half:.0f} s, then block voices)")
        print(window._diagnostics_text())
        verdict = "OK" if diagnostics["underruns"] == 0 and diagnostics["deadline_misses"] == 0 else "GLITCHES"
        print(f"Result: {verdict} — underruns {diagnostics['underruns']}, deadline misses {diagnostics['deadline_misses']}")
        window._stop()
        window.close()
        app.quit()

    QTimer.singleShot(int(half * 1000), switch)
    QTimer.singleShot(int(args.seconds * 1000), finish)
    app.exec()
    os.environ.pop("SONICLAB_BENCH", None)
    print(f"(wall time {time.monotonic() - began:.1f} s)")


if __name__ == "__main__":
    main()
