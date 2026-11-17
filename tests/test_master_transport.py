"""One master clock per song: alignment measured in the audio output.

Every stem is a sine at its own frequency. Fitting each sine's phase in the
mixed output tells which sample of its file a stem plays at each moment, to
a fraction of a sample. Against the master clock (the transport position,
or a song's block clock) that is the stem's misalignment, printed in samples
and milliseconds.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_master_transport -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import stem_block, stem_group
from tests.test_stem_sync import SharedClockWindowTests

BLOCK = 512
FREQS = (440.0, 550.0, 660.0, 770.0, 880.0, 990.0)
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
LEVEL = 0.1  # six at once stay under full scale, so the limiter never acts
TOLERANCE = 0.25  # samples


def sine(freq: float, seconds: float = 8.0) -> np.ndarray:
    frames = np.arange(int(seconds * OUTPUT_RATE))
    tone = (LEVEL * np.sin(2 * np.pi * freq * frames / OUTPUT_RATE)).astype(np.float32)
    return np.column_stack((tone, tone))


def lags(mixed: np.ndarray, starts: dict, freqs: dict) -> dict:
    """``name -> samples`` each sine is late by (positive) or early, where
    ``starts[name]`` is the file frame it should play at the first sample
    of ``mixed``. All sines are fitted together, so none leaks into another."""
    n = np.arange(len(mixed))
    names = list(freqs)
    basis = []
    for name in names:
        w = 2 * np.pi * freqs[name] / OUTPUT_RATE
        basis += [np.sin(w * n), np.cos(w * n)]
    coef, *_ = np.linalg.lstsq(np.stack(basis, axis=1), mixed.astype(np.float64), rcond=None)
    found = {}
    for k, name in enumerate(names):
        s, c = coef[2 * k], coef[2 * k + 1]
        if np.hypot(s, c) < 0.3 * LEVEL:
            found[name] = None  # not sounding
            continue
        w = 2 * np.pi * freqs[name] / OUTPUT_RATE
        # LEVEL·sin(w·(start + n − lag)) = s·sin(w·n) + c·cos(w·n)
        phase = np.arctan2(c, s)
        expected = w * starts[name]
        lag = -((phase - expected + np.pi) % (2 * np.pi) - np.pi) / w
        found[name] = float(lag)
    return found


def report(label: str, found: dict) -> float:
    worst = max(abs(lag) for lag in found.values() if lag is not None)
    print(f"\n  {label}: worst {worst:.3f} samples = {worst / OUTPUT_RATE * 1000:.4f} ms", end="")
    return worst


class TransportTests(unittest.TestCase):
    """The song transport: one sample position for the whole arrangement."""

    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        for freq in FREQS:
            self.engine.add_clip(sine(freq), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.freqs = dict(zip(STEMS, FREQS))

    def render(self, blocks: int) -> tuple[np.ndarray, int]:
        """Mixed output and the transport frame at its first sample."""
        start = int(self.engine._position)
        out = np.zeros((BLOCK, 2), np.float32)
        chunks = []
        for _ in range(blocks):
            self.engine._callback(out, BLOCK, None, None)
            chunks.append(out[:, 0].copy())
        return np.concatenate(chunks), start

    def check(self, label: str, playing=STEMS, starts=None, blocks: int = 8) -> dict:
        mixed, start = self.render(blocks)
        starts = starts or {name: start for name in STEMS}
        found = lags(mixed, starts, self.freqs)
        for name in STEMS:
            if name in playing:
                self.assertIsNotNone(found[name], f"{name} silent ({label})")
            else:
                self.assertIsNone(found[name], f"{name} sounded ({label})")
        self.assertLess(report(label, {k: v for k, v in found.items() if k in playing}), TOLERANCE)
        return found

    def test_six_stems_play_at_one_master_position(self):
        self.engine.play()
        self.check("six stems, transport")

    def test_mute_unmute_and_solo_keep_the_master_position(self):
        self.engine.play()
        self.check("before mute")
        self.engine.set_clip_muted(2, True)
        self.check("other muted", playing=[s for s in STEMS if s != "other"])
        self.engine.set_clip_muted(2, False)  # back at the master position, not its old one
        self.check("other unmuted")
        self.engine.set_clip_solo(4, True)
        self.check("guitar soloed", playing=["guitar"])
        self.engine.set_clip_solo(4, False)
        self.check("solo off")

    def test_a_stem_added_during_playback_joins_the_master_position(self):
        self.engine.clear_clips()
        for freq in FREQS[:5]:
            self.engine.add_clip(sine(freq), 0.0)
        self.engine.play()
        self.check("five stems", playing=STEMS[:5])
        self.engine.add_clip(sine(FREQS[5]), 0.0)  # the sixth arrives mid-play
        self.check("sixth stem added while playing")

    def test_seek_pause_and_resume_move_every_stem(self):
        self.engine.play()
        self.render(10)
        self.engine.seek(3.25)
        self.check("after a seek to 3.25 s")
        self.engine.pause()
        self.render(10)
        self.engine.play()
        self.check("after pause and resume")

    def test_different_starts_and_source_offsets(self):
        # Each stem starts somewhere else on the timeline and from another
        # place in its file: it must still play the file frame the master
        # position gives it.
        places = {}
        for index, name in enumerate(STEMS):
            start, offset = 0.37 * index, 0.11 * index
            self.engine.set_clip_region(index, start, offset, 6.0)
            places[name] = (start, offset)
        self.engine.seek(2.5)
        self.engine.play()
        mixed, at = self.render(8)
        starts = {
            name: int(round(offset * OUTPUT_RATE)) + at - int(round(start * OUTPUT_RATE))
            for name, (start, offset) in places.items()
        }
        found = lags(mixed, starts, self.freqs)
        self.assertLess(report("different starts and offsets", found), TOLERANCE)

    def test_a_shared_loop_wraps_every_stem_on_one_sample(self):
        self.engine.set_loop(1.0, 1.5)
        self.engine.seek(1.4)
        self.engine.play()
        self.render(int(0.1 * OUTPUT_RATE / BLOCK) + 2)  # past the wrap
        self.assertLess(self.engine._position, int(1.5 * OUTPUT_RATE))
        self.check("after the loop wrapped", blocks=6)


class SongClockTests(SharedClockWindowTests):
    """Blocks: one clock per source song, in the audio engine's frames."""

    def make(self, name: str, freqs=FREQS):
        stems = {}
        for stem, freq in zip(STEMS, freqs):
            path = os.path.join(self.root, f"{name}-{stem}.wav")
            sf.write(path, sine(freq), OUTPUT_RATE, subtype="FLOAT")
            stems[stem] = path
        added = self.window._add_stem_tracks(name, stems, STEMS)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        return added

    def render(self, blocks: int) -> tuple[np.ndarray, dict]:
        """Mixed output, and each song clock's frame at its first sample."""
        clocks = {key: group.cursor for key, group in self.engine._groups.items()}
        out = np.zeros((BLOCK, 2), np.float32)
        chunks = []
        for _ in range(blocks):
            self.engine._callback(out, BLOCK, None, None)
            chunks.append(out[:, 0].copy())
        return np.concatenate(chunks), clocks

    def measure(self, label: str, tracks, freqs=FREQS, blocks: int = 8) -> dict:
        """Lag of every playing stem against its song's clock: the file
        frame it plays must be the clock's frame plus its block's shift."""
        mixed, clocks = self.render(blocks)
        starts, wanted = {}, {}
        for track, freq in zip(tracks, freqs):
            index = self.index(track)
            for key, voice in self.engine._voices.items():
                if key[0] == index and voice.playing:
                    starts[track["name"]] = clocks[voice.group] + voice.shift
                    wanted[track["name"]] = freq
        found = lags(mixed, starts, wanted)
        self.assertTrue(all(lag is not None for lag in found.values()), found)
        self.assertLess(report(label, found), TOLERANCE)
        return found

    def test_six_stems_on_different_bars_share_the_song_clock(self):
        tracks = self.make("Song")
        for bar, track in enumerate(tracks):
            self.press(self.index(track), bar % 4)
        self.assertEqual(len({voice.group for voice in self.engine._voices.values()}), 1)
        self.measure("six stems, six bars, one song clock", tracks)

    def test_unmuting_and_adding_a_stem_land_on_the_song_position(self):
        tracks = self.make("Song")
        for track in tracks[:5]:
            self.press(self.index(track), 1)
        self.mix(20)
        self.window._set_muted(self.index(tracks[2]), True)
        self.mix(20)
        self.window._set_muted(self.index(tracks[2]), False)
        self.measure("other unmuted mid-play", tracks[:5], FREQS[:5])
        self.press(self.index(tracks[5]), 1)  # piano joins now
        self.measure("piano added mid-play", tracks, FREQS)
        self.window._set_solo(self.index(tracks[0]), True)
        mixed, _clocks = self.render(8)
        self.assertEqual(lags(mixed, {"x": 0}, {"x": FREQS[1]})["x"], None)  # bass silent under solo
        self.window._set_solo(self.index(tracks[0]), False)
        self.measure("solo off again", tracks, FREQS)

    def test_pause_and_resume_the_whole_song(self):
        tracks = self.make("Song")
        for track in tracks:
            self.press(self.index(track), 2)
        self.mix(10)
        group = stem_group(tracks[0])
        for track in tracks:
            self.press(self.index(track), 2)
        held = self.engine._groups[group].cursor
        self.mix(30)
        self.assertEqual(self.engine._groups[group].cursor, held)
        for track in tracks:
            self.press(self.index(track), 2)
        self.measure("resumed whole song", tracks)

    def test_short_pieces_and_cached_stems_stay_on_one_clock(self):
        fresh = self.make("Song")
        stems = {name: track["path"] for name, track in zip(STEMS, fresh)}
        self.window._finish_split_stems("Song", stems, STEMS, None)  # cached: the same files
        self.app.processEvents()
        cached = self.window.project.tracks[len(fresh):]
        self.assertEqual(stem_group(cached[0]), stem_group(fresh[0]))
        # Cut the cached bass into 0.125 s pieces; play one of them, plus
        # fresh drums and cached vocals.
        bass = cached[1]
        for _ in range(4):
            pieces = [i for i, t in enumerate(self.window.project.tracks) if t is bass or (
                t.get("path") == bass["path"] and t.get("stem_group") == bass["stem_group"] and float(t["start"]) < 1.0
                and float(t["duration"]) > 0.13)]
            for i in sorted(pieces, key=lambda i: -float(self.window.project.tracks[i]["start"])):
                t = self.window.project.tracks[i]
                self.window._split_stem_in_half(i, 0.0, min(float(t["duration"]), 1.0 - float(t["start"])))
        tracks = self.window.project.tracks
        piece = next(t for t in tracks if t.get("path") == bass["path"] and abs(float(t["start"]) - 0.25) < 1e-6
                     and abs(float(t["duration"]) - 0.125) < 1e-6 and t is not fresh[1])
        self.press(self.index(fresh[0]), 0)
        index = self.index(piece)
        self.press(index, self.window._view_for(index).block_at_time(index, 0.26))
        self.press(self.index(cached[3]), 1)
        self.measure("fresh, cached and a 0.125 s piece", [fresh[0], piece, cached[3]], (FREQS[0], FREQS[1], FREQS[3]))

    def test_a_canvas_transfer_and_save_load_keep_alignment(self):
        tracks = self.make("Song")
        self.press(self.index(tracks[0]), 0)
        self.press(self.index(tracks[1]), 0)
        self.mix(10)
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(self.index(tracks[1]), 1, start=3.0)
        self.app.processEvents()
        self.measure("after a canvas transfer", tracks[:2], FREQS[:2])
        path = os.path.join(self.root, "session.soniclab")
        self.window.current_path = path
        self.window._save_project()
        self.window._load_project(path)
        self.app.processEvents()
        loaded = self.window.project.tracks
        self.assertEqual(len({stem_group(t) for t in loaded}), 1)
        self.press(0, 0)
        index = next(i for i, t in enumerate(loaded) if t["name"].endswith("bass"))
        self.press(index, self.window._view_for(index).block_at_time(index, 3.5))
        self.measure("after save and load", [loaded[0], loaded[index]], FREQS[:2])

    def test_different_songs_keep_separate_clocks(self):
        one = self.make("One", FREQS[:6])
        two = self.make("Two", tuple(f + 55.0 for f in FREQS))
        self.assertNotEqual(stem_group(one[0]), stem_group(two[0]))
        self.press(self.index(one[0]), 2)
        self.mix(30)
        first = self.engine._groups[stem_group(one[0])].cursor
        self.press(self.index(two[0]), 0)  # starts at its own block, not at song one's place
        self.assertEqual(self.engine._groups[stem_group(two[0])].cursor, 0)
        self.assertGreater(first, 0)
        self.press(self.index(one[0]), 2)  # pausing song one leaves song two running
        held = self.engine._groups[stem_group(one[0])].cursor
        self.mix(10)
        self.assertEqual(self.engine._groups[stem_group(one[0])].cursor, held)  # song one holds
        self.assertTrue(self.engine.previews()[(self.index(two[0]), stem_block(0))][0])
        self.assertEqual(self.engine._groups[stem_group(two[0])].cursor, 10 * BLOCK)  # song two moves
        self.measure("song two alone", [two[0]], (FREQS[0] + 55.0,))


# The inherited shared-clock tests already run from test_stem_sync.
for _name in [name for name in vars(SharedClockWindowTests) if name.startswith("test_")]:
    setattr(SongClockTests, _name, None)
del SharedClockWindowTests


if __name__ == "__main__":
    unittest.main()
