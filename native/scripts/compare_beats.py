#!/usr/bin/env python3
"""Compare SonicLab's native beat tracker with the DAW's Madmom analysis.

Madmom (the DAW's reference) and the native tracker (the plugins') are run
on the same files. Run from the repository root with the app's interpreter:

    .venv/bin/python native/scripts/compare_beats.py song.wav other.mp3 ...
    .venv/bin/python native/scripts/compare_beats.py --plugin song.wav ...

``--live`` runs the bare tracker in its live mode (fast; same results as
the plugin's live path). ``--plugin`` runs each song through the real plugin instead of the bare
tracker (``SonicLabBeatAnalyzerPluginTests``, linux-juce build): once as
a dropped file, once played live on a stereo bus and once on a mono bus.
Songs are converted to WAV with ffmpeg first, because JUCE on Linux reads
no MP3.

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
import os
import subprocess
import sys
import tempfile
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


def find_plugin_tests() -> Path:
    for preset in ("linux-juce", "macos-xcode/Release", "macos-ninja"):
        candidate = ROOT / "native" / "build" / preset / "plugins" / "beat_analyzer" / "juce" / "SonicLabBeatAnalyzerPluginTests"
        if candidate.is_file():
            return candidate
    raise SystemExit("SonicLabBeatAnalyzerPluginTests not found; build the linux-juce preset first")


def run_plugin(tests: Path, path: Path, work: Path) -> dict:
    """The song through the plugin: {"file": ..., "stereo": ..., "mono": ...}."""
    wav = work / "input.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)], check=True)
    out = work / "result.json"
    environment = dict(os.environ, SONICLAB_REAL_AUDIO=str(wav), SONICLAB_REAL_AUDIO_OUT=str(out))
    done = subprocess.run([str(tests), "plugin.real"], env=environment, capture_output=True, text=True)
    if done.returncode != 0 or not out.is_file():
        raise RuntimeError(f"plugin run failed: {done.stdout[-500:]}")
    return json.loads(out.read_text())


def run_native(cli: Path, path: Path, live: bool = False) -> dict:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(audio.mean(axis=1), dtype="<f4")
    command = [str(cli), str(rate)] + (["60", "200", "0", "live"] if live else [])
    done = subprocess.run(command, input=mono.tobytes(), capture_output=True, check=True)
    return json.loads(done.stdout)


def score(native: dict, ref_beats, ref_down, ref_bpm, from_seconds: float = 0.0) -> tuple:
    beats = [time for time, _ in native["beats"] if time >= from_seconds]
    down = [time for time, position in native["beats"] if position == 1 and time >= from_seconds]
    ref_beats = [time for time in ref_beats if time >= from_seconds]
    ref_down = [time for time in ref_down if time >= from_seconds]
    return (
        native["bpm"],
        tempo_relation(native["bpm"], ref_bpm),
        f_measure(beats, ref_beats),
        f_measure(down, ref_down),
        native["beats_per_bar"],
        native["tempo_confidence"],
    )


def main(arguments: list[str]) -> int:
    plugin = "--plugin" in arguments
    live = "--live" in arguments
    paths = [argument for argument in arguments if argument not in ("--plugin", "--live")]
    if not paths:
        print(__doc__)
        return 2
    runner = find_plugin_tests() if plugin else find_cli()
    client = BeatAnalysisClient()
    modes = ("file", "stereo", "mono") if plugin else ("live",) if live else ("tracker",)
    rows = []
    for name in paths:
        path = Path(name)
        reference = client.analyze_path(path, timeout=900)
        if reference.status != STATUS_OK:
            print(f"{path.name}: Madmom {reference.status}: {reference.message}")
            continue
        ref_beats = [event.time_seconds for event in reference.events]
        ref_down = [event.time_seconds for event in reference.events if event.beat_position == 1]
        ref_bpm = reference_tempo(ref_beats)
        ref_meter = max((event.beat_position for event in reference.events), default=0)
        if plugin:
            with tempfile.TemporaryDirectory() as work:
                results = run_plugin(runner, path, Path(work))
        elif live:
            results = {"live": run_native(runner, path, live=True)}
        else:
            results = {"tracker": run_native(runner, path)}
        for mode in modes:
            # Live tracking needs a few seconds to lock on; score it after 8 s.
            start = 8.0 if mode in ("stereo", "mono", "live") else 0.0
            rows.append((path.name[:30], mode, ref_bpm, ref_meter) + score(results[mode], ref_beats, ref_down, ref_bpm, start))
    print(f"{'file':30} {'mode':7} {'native':>7} {'madmom':>7} {'tempo':9} {'beat F':>6} {'down F':>6} {'meter':>5} {'conf':>5}")
    for name, mode, ref_bpm, ref_meter, bpm, relation, beat_f, down_f, meter, confidence in rows:
        print(f"{name:30} {mode:7} {bpm:7.1f} {ref_bpm:7.1f} {relation:9} {beat_f:6.2f} {down_f:6.2f} {meter:>2}/{ref_meter:<2} {confidence:5.2f}")
    for mode in modes:
        chosen = [row for row in rows if row[1] == mode]
        if chosen:
            same = sum(row[5] == "same" for row in chosen)
            print(f"{mode:7} tempo agrees on {same}/{len(chosen)}; mean beat F {np.mean([row[6] for row in chosen]):.2f}; "
                  f"mean downbeat F {np.mean([row[7] for row in chosen]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
