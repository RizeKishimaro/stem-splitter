#!/usr/bin/env python3
"""Check that the installed SonicLab environment can import what it needs."""

from __future__ import annotations

import json
import sys

REQUIRED = ("PySide6", "numpy", "scipy", "soundfile", "sounddevice")
OPTIONAL = ("pyqtgraph", "aubio")


def main() -> int:
    report: dict[str, object] = {}
    failed = False
    for name in REQUIRED + OPTIONAL:
        try:
            __import__(name)
            report[name] = True
        except Exception as exc:
            report[name] = f"{type(exc).__name__}: {exc}"
            if name in REQUIRED:
                failed = True
    try:
        import soniclab.main  # noqa: F401

        report["soniclab"] = True
    except Exception as exc:
        report["soniclab"] = f"{type(exc).__name__}: {exc}"
        failed = True
    if report.get("sounddevice") is True:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            report["audio_devices"] = len(devices)
        except Exception as exc:
            report["audio_devices"] = f"{type(exc).__name__}: {exc}"
    print("SONICLAB_VERIFY " + json.dumps(report))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
