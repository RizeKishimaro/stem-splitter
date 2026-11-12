"""Downbeat worker backed by madmom-infer.

This speaks the same JSON protocol as ``tools/madmom_worker.py``. SonicLab's
client never imports either library. stdout is one JSON document per request.
Progress goes to stderr as ``stage <name>``.

Timestamps are seconds from the start of the source. ``beat_position`` 1 is
the downbeat. Tempo is the median inter-beat interval, in BPM, matching the
legacy worker. The downbeat tracker does not return a per-beat confidence.

The pretrained weights are not in this package. madmom-infer downloads the
CPJKU downbeat models on first use and caches them. Those bytes are
CC BY-NC-SA 4.0. This worker does not bundle them and does not make them
commercially usable.

``--serve`` keeps the process alive and the networks loaded. Each stdin line
is one JSON request, and each stdout line is one JSON result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
from pathlib import Path

SCHEMA_VERSION = 1
WORKER_VERSION = "1"
TIME_UNIT = "seconds"
BEAT_POSITION_SEMANTICS = "1_is_downbeat"
TEMPO_UNIT = "bpm"
ENGINE = "madmom"
MODEL_RATE = 44100

_rnn = None
_trackers: dict[tuple[int, tuple[int, ...]], object] = {}


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


def _wav_rate(path: str) -> int | None:
    try:
        from scipy.io import wavfile
    except ImportError:
        return None
    try:
        rate, _audio = wavfile.read(path)
    except Exception:
        return None
    try:
        return int(rate)
    except (TypeError, ValueError):
        return None


def _write_wav(path: Path, audio, rate: int) -> None:
    from scipy.io import wavfile
    import numpy as np

    array = np.asarray(audio, dtype=np.float32)
    wavfile.write(str(path), int(rate), array)


def _resample_wav(source: str, dest: Path, rate: int) -> None:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    import numpy as np

    file_rate, audio = wavfile.read(source)
    file_rate = int(file_rate)
    array = np.asarray(audio)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32, copy=False)
    if file_rate != MODEL_RATE:
        factor = gcd(file_rate, MODEL_RATE)
        array = resample_poly(array, MODEL_RATE // factor, file_rate // factor, axis=0)
        array = np.asarray(array, dtype=np.float32)
    _write_wav(dest, array, MODEL_RATE)


def _ffmpeg_to_wav(source: str, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "This audio is not a 44100 Hz WAV, and ffmpeg is not available to decode it."
        )
    done = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            source,
            "-ar",
            str(MODEL_RATE),
            "-f",
            "wav",
            "-acodec",
            "pcm_f32le",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or not dest.is_file():
        detail = (done.stderr or done.stdout or "").strip()
        raise RuntimeError(detail or f"ffmpeg could not decode {source}")


def _prepare(path: str) -> tuple[str, list[Path]]:
    """Give the processor a 44100 Hz WAV. madmom-infer does not resample."""
    temps: list[Path] = []
    rate = _wav_rate(path)
    if rate == MODEL_RATE:
        return path, temps
    descriptor, name = tempfile.mkstemp(prefix="soniclab-infer-", suffix=".wav")
    import os

    os.close(descriptor)
    dest = Path(name)
    temps.append(dest)
    if rate is not None:
        _resample_wav(path, dest, rate)
    else:
        _ffmpeg_to_wav(path, dest)
    return str(dest), temps


def _processors(fps: int, beats_per_bar: list[int]):
    global _rnn
    from madmom_infer.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    if _rnn is None:
        _rnn = RNNDownBeatProcessor()
    key = (int(fps), tuple(int(bar) for bar in beats_per_bar))
    tracker = _trackers.get(key)
    if tracker is None:
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=list(key[1]), fps=key[0])
        _trackers[key] = tracker
    return _rnn, tracker


def _explain(exc: BaseException) -> str:
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return f"Beat model download failed: {exc}"
    text = str(exc)
    if "sha256" in text or "downloaded model" in text:
        return f"Beat model download failed: {text}"
    return text


def analyze(path: str, fps: int, beats_per_bar: list[int]) -> dict:
    import madmom_infer

    version = f"madmom-infer {getattr(madmom_infer, '__version__', '')}".strip()
    _stage("loading")
    prepared, temps = _prepare(path)
    try:
        rnn, tracker = _processors(fps, beats_per_bar)
        _stage("detecting")
        activations = rnn(prepared)
        _stage("tracking")
        tracked = tracker(activations)
        events = _events(tracked)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)
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


def _run_request(path: str, fps: int, bars: list[int]) -> dict:
    if fps <= 0:
        return _payload("error", "fps must be positive", fps=fps, beats_per_bar=bars)
    if any(bar < 1 for bar in bars):
        return _payload("error", "beats per bar must be positive", fps=fps, beats_per_bar=bars)
    if not Path(path).is_file():
        return _payload("error", f"audio file not found: {path}", fps=fps, beats_per_bar=bars)
    try:
        return analyze(path, fps, bars)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _payload("error", _explain(exc), fps=fps, beats_per_bar=bars)


def _serve() -> int:
    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            return 0
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except json.JSONDecodeError as exc:
            _emit(_payload("error", f"request JSON is invalid: {exc}"))
            continue
        if not isinstance(request, dict):
            _emit(_payload("error", "request JSON must be an object"))
            continue
        if request.get("cmd") == "quit":
            return 0
        path = str(request.get("audio") or "")
        try:
            fps = int(request.get("fps"))
            bars = [int(bar) for bar in request.get("bars") or []]
        except (TypeError, ValueError):
            _emit(_payload("error", "fps and bars must be integers"))
            continue
        document = _run_request(path, fps, bars)
        _emit(document)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect beats and downbeats with madmom-infer")
    parser.add_argument("audio", nargs="?", help="Existing audio file. It is read in place.")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--bars", type=int, nargs="+")
    parser.add_argument("--serve", action="store_true", help="Read JSON requests from stdin and keep models loaded")
    args = parser.parse_args(argv)
    if args.serve:
        return _serve()
    if not args.audio or args.fps is None or not args.bars:
        _emit(_payload("error", "audio, --fps, and --bars are required"))
        return 1
    document = _run_request(args.audio, args.fps, list(args.bars))
    _emit(document)
    return 0 if document["status"] in ("ok", "empty") else 1


if __name__ == "__main__":
    sys.exit(main())
