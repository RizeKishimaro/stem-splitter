"""Downbeat worker for the isolated Python 3.11 Madmom environment.

Run this file with ``.venv-madmom/bin/python``. The SonicLab application is
Python 3.12 and must not import it.

stdout is one JSON document. Progress and library warnings go to stderr.
Timestamps are seconds from the start of the source audio. ``beat_position``
uses musical counting: 1 is the downbeat, then 2, 3, and so on inside the bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

SCHEMA_VERSION = 1
WORKER_VERSION = "1"
TIME_UNIT = "seconds"
BEAT_POSITION_SEMANTICS = "1_is_downbeat"
TEMPO_UNIT = "bpm"
ENGINE = "madmom"


def _stage(name: str) -> None:
    print(f"stage {name}", file=sys.stderr, flush=True)


def _payload(status: str, message: str = "", **extra) -> dict:
    document = {
        "schema_version": SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "engine": ENGINE,
        "engine_version": extra.pop("engine_version", None),
        "status": status,
        "message": message,
        "time_unit": TIME_UNIT,
        "beat_position_semantics": BEAT_POSITION_SEMANTICS,
        "tempo_unit": TEMPO_UNIT,
        "tempo_bpm": None,
        "fps": None,
        "beats_per_bar": [],
        "sample_rate": None,
        "events": [],
    }
    document.update(extra)
    return document


def _emit(document: dict) -> None:
    json.dump(document, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _events(tracked) -> list[dict]:
    events = []
    previous = None
    for row in tracked:
        seconds = float(row[0])
        position = float(row[1])
        if seconds != seconds or seconds in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite beat time: {seconds}")
        if seconds < 0.0:
            raise ValueError(f"negative beat time: {seconds}")
        if previous is not None and seconds <= previous:
            raise ValueError("beat times are not strictly increasing")
        nearest = round(position)
        if abs(position - nearest) > 1e-6 or nearest < 1:
            raise ValueError(f"beat position is not a positive integer: {position}")
        events.append({"time_seconds": seconds, "beat_position": int(nearest)})
        previous = seconds
    return events


def _tempo(events: list[dict]) -> float | None:
    if len(events) < 2:
        return None
    intervals = [
        events[index]["time_seconds"] - events[index - 1]["time_seconds"]
        for index in range(1, len(events))
    ]
    intervals.sort()
    mid = len(intervals) // 2
    if len(intervals) % 2:
        median = intervals[mid]
    else:
        median = (intervals[mid - 1] + intervals[mid]) / 2.0
    if median <= 0.0:
        return None
    return 60.0 / median


def analyze(path: str, fps: int, beats_per_bar: list[int]) -> dict:
    _stage("loading")
    import madmom
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    version = str(getattr(madmom, "__version__", "") or "")
    _stage("detecting")
    activations = RNNDownBeatProcessor()(path)
    _stage("tracking")
    tracked = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)(activations)
    events = _events(tracked)
    _stage("done")
    status = "ok" if events else "empty"
    return _payload(
        status,
        engine_version=version,
        tempo_bpm=_tempo(events),
        fps=int(fps),
        beats_per_bar=[int(bar) for bar in beats_per_bar],
        events=events,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect beats and downbeats with Madmom")
    parser.add_argument("audio", help="Existing audio file. It is read in place.")
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--bars", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        _emit(_payload("error", "fps must be positive", fps=args.fps))
        return 1
    if any(bar < 1 for bar in args.bars):
        _emit(_payload("error", "beats per bar must be positive", beats_per_bar=args.bars))
        return 1
    try:
        document = analyze(args.audio, args.fps, list(args.bars))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit(_payload("error", str(exc), fps=args.fps, beats_per_bar=list(args.bars)))
        return 1
    _emit(document)
    return 0 if document["status"] in ("ok", "empty") else 1


if __name__ == "__main__":
    sys.exit(main())
