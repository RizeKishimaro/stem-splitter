#!/usr/bin/env python3
"""Compare SonicLab's native beat tracker with the DAW's Madmom analysis.

Madmom (the DAW's reference) and the native tracker (the plugins') are run
on the same files. Run from the repository root with the app's interpreter:

    .venv/bin/python native/scripts/compare_beats.py song.wav other.mp3 ...

Needs the native build (``cmake --build --preset linux-release``) and the
Madmom worker environment. Reported per file:

    tempo      native BPM against Madmom's, and whether they agree within
               4 %, or are related by 2x or 3x (an "octave" reading)
    beats F    beat F-measure against Madmom's beats, +-70 ms
    downbeat F the same for downbeats
    meter      beats per bar from each
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from soniclab.core.beat_analysis import STATUS_OK, BeatAnalysisClient  # noqa: E402

TOLERANCE = 0.07


def find_cli() -> Path:
    for preset in ("linux-release", "linux-debug", "linux-juce", "macos-ninja"):
        candidate = ROOT / "native" / "build" / preset / "plugins" / "beat_analyzer" / "soniclab_beat_cli"
        if candidate.is_file():
            return candidate
    raise SystemExit("soniclab_beat_cli not found; build native/ first")


def f_measure(detected, reference) -> float:
    if not len(detected) or not len(reference):
        return 0.0
    used = np.zeros(len(reference), dtype=bool)
    hits = 0
    for time in detected:
        distance = np.abs(np.asarray(reference) - time)
        distance[used] = np.inf
        nearest = int(np.argmin(distance))
        if distance[nearest] <= TOLERANCE:
            used[nearest] = True
            hits += 1
    precision = hits / len(detected)
    recall = hits / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tempo_relation(native: float, reference: float) -> str:
    if native <= 0 or reference <= 0:
        return "none"
    for name, ratio in (("same", 1.0), ("half", 0.5), ("double", 2.0), ("third", 1 / 3), ("triple", 3.0), ("2/3", 2 / 3), ("3/2", 1.5)):
        if abs(native / reference - ratio) <= 0.04 * ratio:
            return name
    return "different"


def reference_tempo(beats) -> float:
    """Tempo from the mean beat interval, ignoring gaps and stumbles.

    Madmom's beats sit on a 10 ms grid, so the median interval is too coarse
    (0.48 s is 125 BPM, 0.47 s is 127.7).
    """
    if len(beats) < 3:
        return 0.0
    intervals = np.diff(beats)
    median = np.median(intervals)
    regular = intervals[np.abs(intervals - median) < 0.15 * median]
    return 60.0 / float(regular.mean()) if regular.size else 0.0


def run_native(cli: Path, path: Path) -> dict:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(audio.mean(axis=1), dtype="<f4")
    done = subprocess.run([str(cli), str(rate)], input=mono.tobytes(), capture_output=True, check=True)
    return json.loads(done.stdout)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    cli = find_cli()
    client = BeatAnalysisClient()
    rows = []
    for name in paths:
        path = Path(name)
        native = run_native(cli, path)
        reference = client.analyze_path(path, timeout=900)
        if reference.status != STATUS_OK:
            print(f"{path.name}: Madmom {reference.status}: {reference.message}")
            continue
        ref_beats = [event.time_seconds for event in reference.events]
        ref_down = [event.time_seconds for event in reference.events if event.beat_position == 1]
        ref_bpm = reference_tempo(ref_beats)
        ref_meter = max((event.beat_position for event in reference.events), default=0)
        beats = [time for time, _ in native["beats"]]
        down = [time for time, position in native["beats"] if position == 1]
        rows.append(
            (
                path.name[:38],
                native["bpm"],
                ref_bpm,
                tempo_relation(native["bpm"], ref_bpm),
                f_measure(beats, ref_beats),
                f_measure(down, ref_down),
                native["beats_per_bar"],
                ref_meter,
                native["tempo_confidence"],
            )
        )
    print(f"{'file':38}  {'native':>7} {'madmom':>7} {'tempo':9} {'beat F':>6} {'down F':>6} {'meter':>5} {'conf':>5}")
    for name, bpm, ref_bpm, relation, beat_f, down_f, meter, ref_meter, confidence in rows:
        print(f"{name:38}  {bpm:7.1f} {ref_bpm:7.1f} {relation:9} {beat_f:6.2f} {down_f:6.2f} {meter:>2}/{ref_meter:<2} {confidence:5.2f}")
    if rows:
        same = sum(row[3] == "same" for row in rows)
        print(f"\ntempo agrees on {same}/{len(rows)}; mean beat F {np.mean([row[4] for row in rows]):.2f}; "
              f"mean downbeat F {np.mean([row[5] for row in rows]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
