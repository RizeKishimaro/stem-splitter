"""Stems of one split share a playback clock: later stems join where the
others are, never at zero, and stay sample-aligned through pause, stop,
mute, canvas transfers and save/load. Unrelated splits keep their own.

Alignment is measured from the output itself. Stem k plays a ramp scaled
by (k + 1), so file frame n of stem k is (k + 1) * n * STEP. With every
playing stem on the same frame the mix is (sum of scales) * n * STEP; a
stem d samples late lowers it by (k + 1) * d * STEP, so the error in
samples is read straight off the mix.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import LIMIT_CEILING, OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import MainWindow, stem_block, stem_group

SECONDS = 8
STEP = 1.0 / (OUTPUT_RATE * 400)  # ramp height per sample: 0.02 after 8 s
BLOCK = 512


def ramp(scale: float, seconds: float = SECONDS) -> np.ndarray:
    line = np.arange(int(OUTPUT_RATE * seconds), dtype=np.float64) * STEP * scale
    return np.column_stack((line, line)).astype(np.float32)


def run(engine, blocks: int, frames: int = BLOCK) -> np.ndarray:
    out = np.zeros((frames, 2), np.float32)
    mixed = []
    for _ in range(blocks):
        engine._callback(out, frames, None, None)
        mixed.append(out[:, 0].copy())
    return np.concatenate(mixed) if mixed else np.zeros(0, np.float32)


def lag_samples(mixed, frame, scales, shifts=None) -> float:
    """Worst lag in samples of the playing stems, given the mix and the
    shared frame its first sample should be at. ``shifts`` (frames, one per
    stem) are stems playing another place of their file on that clock."""
    shifts = [0] * len(scales) if shifts is None else shifts
    at = frame + np.arange(len(mixed))
    expected = sum((at + shift) * scale for scale, shift in zip(scales, shifts)) * STEP
    return float(np.max(np.abs(expected - mixed)) / (STEP * min(scales)))


def report(name, samples):
    print(f"\n  {name}: sync error {samples:.3f} samples = {samples / OUTPUT_RATE * 1000:.4f} ms", end="")


class SharedClockEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        for scale in (1, 2, 3, 4):
            self.engine.add_clip(ramp(scale), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def cursor(self, group="song"):
        return self.engine._groups[group].cursor

    def test_a_late_stem_joins_at_the_shared_position_on_the_same_sample(self):
        for frames in (BLOCK, 441, 1024):
            with self.subTest(frames=frames):
                self.engine.preview_stop_all()
                self.engine.stem_play("song", [("a", 0)], 1.0)
                run(self.engine, 7, frames)
                joined_at = self.cursor()
                self.assertEqual(joined_at, OUTPUT_RATE + 7 * frames)
                self.engine.stem_play("song", [("b", 1)], 0.0)  # start is ignored: it joins
                mixed = run(self.engine, 20, frames)
                error = lag_samples(mixed, joined_at, (1, 2))
                report(f"late join, {frames}-frame blocks", error)
                self.assertLess(error, 0.25)

    def test_pause_resume_and_stop_leave_the_others_alone(self):
        self.engine.stem_play("song", [("a", 0), ("b", 1)], 0.0)
        run(self.engine, 4)
        self.engine.preview_pause("a")
        frame = self.cursor()
        alone = run(self.engine, 6)
        self.assertLess(lag_samples(alone, frame, (2,)), 0.25)  # b went on, unaffected
        self.engine.stem_play("song", [("a", 0)])  # resumes at the shared place
        frame = self.cursor()
        self.assertEqual(frame, 10 * BLOCK)
        both = run(self.engine, 6)
        error = lag_samples(both, frame, (1, 2))
        report("resume after pause", error)
        self.assertLess(error, 0.25)
        self.engine.preview_stop("b")
        frame = self.cursor()
        self.assertLess(lag_samples(run(self.engine, 3), frame, (1,)), 0.25)
        self.engine.stem_play("song", [("b", 1)])  # a stopped stem joins again, not at zero
        frame = self.cursor()
        self.assertEqual(frame, 19 * BLOCK)
        self.assertLess(lag_samples(run(self.engine, 3), frame, (1, 2)), 0.25)

    def test_the_clock_holds_while_every_stem_is_paused(self):
        self.engine.stem_play("song", [("a", 0), ("b", 1)], 0.0)
        run(self.engine, 3)
        self.engine.preview_pause("a")
        self.engine.preview_pause("b")
        held = self.cursor()
        self.assertFalse(run(self.engine, 5).any())
        self.assertEqual(self.cursor(), held)
        self.engine.stem_play("song", [("b", 1)])
        self.assertLess(lag_samples(run(self.engine, 2), held, (2,)), 0.25)

    def test_the_last_stop_ends_the_session(self):
        self.engine.stem_play("song", [("a", 0)], 2.0)
        run(self.engine, 3)
        self.engine.preview_stop("a")
        self.assertNotIn("song", self.engine._groups)
        self.assertIsNone(self.engine.group_position("song"))
        self.engine.stem_play("song", [("b", 1)], 0.5)  # a new session starts where asked
        self.assertEqual(self.cursor(), OUTPUT_RATE // 2)

    def test_source_offsets_and_arrangement_places_are_kept(self):
        # b is trimmed to file seconds 3..5 and sits at 6.0 s in the
        # arrangement (moved, as on another canvas); a sits at 0.
        self.engine.set_clip_region(1, 6.0, 3.0, 2.0)
        self.engine.stem_play("song", [("a", 0), ("b", 1)], 2.5)
        early = run(self.engine, 10)  # file 2.5 s: b's region has not begun
        self.assertLess(lag_samples(early, int(2.5 * OUTPUT_RATE), (1,)), 0.25)
        while self.cursor() < 3 * OUTPUT_RATE:
            run(self.engine, 1)
        frame = self.cursor()
        mixed = run(self.engine, 20)
        error = lag_samples(mixed, frame, (1, 2))
        report("offset 3 s, arrangement 6 s", error)
        self.assertLess(error, 0.25)
        playing, seconds = self.engine.previews()["b"]
        # Shown in arrangement time on b's clip: 6.0 + (file time - 3.0).
        self.assertAlmostEqual(seconds, 6.0 + self.cursor() / OUTPUT_RATE - 3.0, delta=BLOCK / OUTPUT_RATE)
        while "b" in self.engine.previews():
            run(self.engine, 1)
        self.assertGreaterEqual(self.cursor(), 5 * OUTPUT_RATE)  # b ended with its region

    def test_stems_of_different_lengths(self):
        engine = self.engine
        short = engine.add_clip(ramp(5, seconds=1.0), 0.0)
        engine.stem_play("song", [("a", 0), ("short", short)], 0.0)
        mixed = run(engine, OUTPUT_RATE // BLOCK)
        self.assertLess(lag_samples(mixed, 0, (1, 5)), 0.25)
        while "short" in engine.previews():
            run(engine, 1)
        frame = self.cursor()
        self.assertLess(lag_samples(run(engine, 10), frame, (1,)), 0.25)  # a plays on alone
        self.assertIn("a", engine.previews())

    def test_separate_splits_keep_separate_clocks(self):
        self.engine.stem_play("one", [("a", 0)], 0.0)
        run(self.engine, 5)
        self.engine.stem_play("two", [("c", 2)], 3.0)
        self.assertEqual(self.engine._groups["two"].cursor, 3 * OUTPUT_RATE)
        run(self.engine, 4)
        self.engine.preview_pause("a")
        run(self.engine, 4)
        self.assertEqual(self.engine._groups["one"].cursor, 9 * BLOCK)
        self.assertEqual(self.engine._groups["two"].cursor, 3 * OUTPUT_RATE + 8 * BLOCK)
        self.engine.preview_stop("c")
        self.assertIn("one", self.engine._groups)

    def test_a_muted_stem_keeps_its_place_on_the_clock(self):
        self.engine.stem_play("song", [("a", 0), ("b", 1, True)], 0.0)
        frame = self.cursor()
        self.assertLess(lag_samples(run(self.engine, 4), frame, (1,)), 0.25)
        self.engine.preview_set_muted("b", False)
        frame = self.cursor()
        self.assertLess(lag_samples(run(self.engine, 4), frame, (1, 2)), 0.25)

    def test_the_transport_is_never_moved(self):
        self.engine.seek(1.5)
        self.engine.stem_play("song", [("a", 0), ("b", 1)], 4.0)
        run(self.engine, 50)
        self.assertFalse(self.engine.playing)
        self.assertEqual(self.engine.position, 1.5)
        self.assertEqual(self.engine._heard, ())

    def test_loud_stems_never_clip(self):
        engine = AudioEngine()
        engine._capture_enabled = False
        for _ in range(4):
            engine.add_clip(np.full((OUTPUT_RATE, 2), 0.9, np.float32), 0.0)
        with patch.object(engine, "_ensure_stream", lambda: None):
            engine.stem_play("song", [(k, k) for k in range(4)], 0.0)
            peaks = [float(np.max(np.abs(run(engine, 1)))) for _ in range(40)]
        self.assertLessEqual(max(peaks), LIMIT_CEILING + 1e-6)
        self.assertGreater(min(peaks), 0.9)  # limited, not silenced

    def test_the_callback_stays_well_inside_its_deadline(self):
        """Eight stems in two splits, plus the song: no block may come
        near the time the device gives it (a dropout)."""
        for scale in (5, 6, 7, 8):
            self.engine.add_clip(ramp(scale), 0.0)
        self.engine.stem_play("one", [(k, k) for k in range(4)], 0.0)
        self.engine.stem_play("two", [(k, k) for k in range(4, 8)], 1.0)
        self.engine._playing = True
        out = np.zeros((BLOCK, 2), np.float32)
        costs = []
        for _ in range(400):
            started = time.perf_counter()
            self.engine._callback(out, BLOCK, None, None)
            costs.append(time.perf_counter() - started)
        self.engine._playing = False
        budget = BLOCK / OUTPUT_RATE
        p99 = float(np.percentile(costs, 99))
        print(f"\n  callback p99 {p99 * 1000:.3f} ms of a {budget * 1000:.2f} ms block", end="")
        self.assertLess(p99, budget / 3)


class _Jobs:
    def events_for(self, path):
        return ()

    def result_for(self, path):
        return None

    def submit(self, path):
        return "missing"

    def poll(self):
        return None

    def cancel(self):
        pass


class SharedClockWindowTests(unittest.TestCase):
    """Through the UI: block ▶ buttons, mixer mute, canvases, save/load."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = self.folder.name
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs()
        self.window.resize(1400, 900)
        self.window.project.bars = [float(s) for s in range(0, SECONDS + 1, 2)]
        self.engine = self.window.engine
        self.engine._capture_enabled = False
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.window.show()

    def tearDown(self):
        self.engine.preview_stop_all()
        for timeline in (self.window.timeline, self.window.timeline2):
            timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def split(self, name, rate=OUTPUT_RATE):
        """Stem files as Demucs would leave them, added like a split."""
        stems = {}
        for scale, stem in enumerate(("drums", "bass", "other", "vocals"), start=1):
            path = os.path.join(self.root, f"{name}-{stem}.wav")
            seconds = np.arange(int(rate * SECONDS)) / rate
            line = (seconds * OUTPUT_RATE * STEP * scale).astype(np.float32)
            sf.write(path, np.column_stack((line, line)), rate, subtype="FLOAT")
            stems[stem] = path
        added = self.window._add_stem_tracks(name, stems)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        return added

    def press(self, track, bar):
        view = self.window._view_for(track)
        view.bar_play_requested.emit(bar, track)

    def index(self, track):
        return next(i for i, t in enumerate(self.window.project.tracks) if t is track)

    def mix(self, blocks):
        mixed = run(self.engine, blocks)
        self.window._animate_playheads()
        return mixed

    def test_one_split_shares_a_group_and_another_does_not(self):
        first = self.split("One")
        second = self.split("Two")
        self.assertEqual(len({stem_group(track) for track in first}), 1)
        self.assertIsNotNone(stem_group(first[0]))
        self.assertNotEqual(stem_group(first[0]), stem_group(second[0]))

    def test_first_play_starts_at_the_block_and_later_stems_join(self):
        drums, bass, other, _vocals = self.split("Song")
        scroll = self.window.timeline.horizontalScrollBar().value()
        self.press(self.index(drums), 1)  # bar 2 starts at 2.0 s
        group = stem_group(drums)
        self.assertEqual(self.engine._groups[group].cursor, 2 * OUTPUT_RATE)
        self.mix(40)
        joined_at = self.engine._groups[group].cursor
        self.press(self.index(bass), 1)  # the same bar joins at the clock
        self.press(self.index(other), 1)
        mixed = self.mix(30)
        error = lag_samples(mixed, joined_at, (1, 2, 3))
        report("window: three stems, two joined late", error)
        self.assertLess(error, 0.25)
        states = self.engine.previews()
        keys = [(self.index(t), stem_block(1)) for t in (drums, bass, other)]
        self.assertTrue(all(states[key][0] for key in keys))
        self.assertEqual(len({round(states[key][1], 6) for key in keys}), 1)
        # The song's transport, its playhead and the scroll stay put.
        self.assertFalse(self.engine.playing)
        self.assertEqual(self.engine.position, 0.0)
        self.assertEqual(self.window.timeline.horizontalScrollBar().value(), scroll)
        # The ⏸ is on the block the clock is in, and toggles that stem only.
        seconds = states[keys[1]][1]
        bar = int(seconds // 2)
        self.press(self.index(bass), bar)
        self.assertFalse(self.engine.previews()[keys[1]][0])
        self.assertTrue(self.engine.previews()[keys[0]][0])

    def test_a_block_loops_with_its_stems_in_sync(self):
        drums, bass, other, vocals = self.split("Song")
        self.press(self.index(drums), 2)  # bar 3: 4.0 s to 6.0 s
        group = stem_group(drums)
        self.mix(20)
        joined_at = self.engine._groups[group].cursor
        self.press(self.index(bass), 2)
        self.assertEqual(self.engine._groups[group].loop, (4 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        wrap = 6 * OUTPUT_RATE - joined_at
        mixed = self.mix((wrap + 2 * 2 * OUTPUT_RATE) // BLOCK + 1)  # two more times round
        error = lag_samples(mixed[:wrap], joined_at, (1, 2))
        fade = 64  # the wrap's crossfade
        cycle = 2 * OUTPUT_RATE
        again = lag_samples(mixed[wrap + fade : wrap + cycle], 4 * OUTPUT_RATE + fade, (1, 2))
        report("window: block loop, a stem joined late", max(error, again))
        self.assertLess(max(error, again), 0.25)
        self.assertTrue(np.array_equal(mixed[wrap : wrap + cycle], mixed[wrap + cycle : wrap + 2 * cycle]))
        keys = [(self.index(t), stem_block(2)) for t in (drums, bass)]
        states = self.engine.previews()
        self.assertTrue(all(states[key][0] for key in keys))  # still playing
        # ⏸ pauses one stem; the other keeps looping.
        self.press(self.index(bass), 2)
        self.assertFalse(self.engine.previews()[keys[1]][0])
        self.assertTrue(self.engine.previews()[keys[0]][0])
        # Picked blocks start together and loop their whole span.
        self.engine.preview_stop_all()
        self.window._view_for(self.index(other)).set_picks({self.index(other): {1}, self.index(vocals): {1, 2}})
        self.window._play_picked_blocks()
        self.assertEqual(self.engine._groups[group].cursor, 2 * OUTPUT_RATE)
        self.assertEqual(self.engine._groups[group].loop, (2 * OUTPUT_RATE, 4 * OUTPUT_RATE))
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertEqual(playing, {(self.index(other), stem_block(1)), (self.index(vocals), stem_block(1)), (self.index(vocals), stem_block(2))})

    def test_stems_on_different_bars_loop_their_own_bars_in_step(self):
        drums, bass, other, _vocals = self.split("Song")
        self.press(self.index(drums), 1)  # bar 2: 2.0 s to 4.0 s
        group = stem_group(drums)
        self.mix(30)
        joined_at = self.engine._groups[group].cursor
        self.press(self.index(bass), 3)  # bar 4: 6.0 s to 8.0 s
        self.press(self.index(other), 0)  # bar 1: 0.0 s to 2.0 s
        cycle = 2 * OUTPUT_RATE
        shifts = (0, 2 * cycle, -cycle)
        wrap = 4 * OUTPUT_RATE - joined_at
        mixed = self.mix((wrap + cycle) // BLOCK + 1)
        error = lag_samples(mixed[:wrap], joined_at, (1, 2, 3), shifts)
        fade = 64  # the wrap's crossfade
        again = lag_samples(mixed[wrap + fade : wrap + cycle], cycle + fade, (1, 2, 3), shifts)
        report("window: three stems on three bars", max(error, again))
        self.assertLess(max(error, again), 0.25)
        # Each pause button shows on the stem's own bar.
        states = self.engine.previews()
        view = self.window._view_for(self.index(bass))
        for track, bar in ((drums, 1), (bass, 3), (other, 0)):
            index = self.index(track)
            self.assertEqual(view.block_at_time(index, states[(index, stem_block(bar))][1]), bar)
        # Another bar of a playing stem loops too, in step with the rest.
        self.press(self.index(bass), 2)
        frame = self.engine._groups[group].cursor
        both = (self.index(bass), stem_block(3)), (self.index(bass), stem_block(2))
        self.assertTrue(all(self.engine.previews()[key][0] for key in both))
        self.assertEqual(self.engine._voices[both[1]].shift, cycle)
        mixed = self.mix(10)
        error = lag_samples(mixed, frame, (1, 2, 2, 3), (0, 2 * cycle, cycle, -cycle))
        report("window: one stem on two bars", error)
        self.assertLess(error, 0.25)
        # Pausing one bar leaves the other playing, and its line stops.
        self.press(self.index(bass), 2)
        paused = self.engine.previews()[both[1]]
        self.assertFalse(paused[0])
        self.assertTrue(self.engine.previews()[both[0]][0])
        self.mix(20)
        self.assertEqual(self.engine.previews(now=time.monotonic() + 0.5)[both[1]], paused)
        self.assertEqual(self.window._view_for(self.index(bass)).block_at_time(self.index(bass), paused[1]), 2)

    def test_mixer_mute_and_volume_apply_to_one_stem(self):
        drums, bass, *_rest = self.split("Song")
        self.press(self.index(drums), 0)
        self.press(self.index(bass), 0)
        self.window._set_muted(self.index(bass), True)
        frame = self.engine._groups[stem_group(drums)].cursor
        self.assertLess(lag_samples(self.mix(4), frame, (1,)), 0.25)
        self.assertIn((self.index(bass), stem_block(0)), self.engine.previews())
        self.window._set_muted(self.index(bass), False)
        self.window._set_gain(self.index(bass), 0.5)  # the fader: bass now counts as 1
        frame = self.engine._groups[stem_group(drums)].cursor
        self.assertLess(lag_samples(self.mix(4), frame, (1, 1)), 0.25)

    def test_a_canvas_transfer_keeps_the_stems_in_sync(self):
        drums, bass, *_rest = self.split("Song")
        self.press(self.index(drums), 0)
        self.press(self.index(bass), 0)
        self.mix(10)
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(self.index(bass), 1, start=3.0)  # and somewhere else
        self.app.processEvents()
        self.mix(2)
        group = stem_group(drums)
        frame = self.engine._groups[group].cursor
        mixed = self.mix(20)
        error = lag_samples(mixed, frame, (1, 2))
        report("after a canvas transfer", error)
        self.assertLess(error, 0.25)
        states = self.engine.previews()
        self.assertTrue(states[(self.index(bass), stem_block(0))][0])
        self.assertFalse(self.engine.playing)

    def test_save_and_load_keep_the_split_together(self):
        self.split("Song")
        path = os.path.join(self.root, "session.soniclab")
        self.window.current_path = path
        self.window._save_project()
        self.window._load_project(path)
        self.app.processEvents()
        tracks = self.window.project.tracks
        self.assertEqual(len({stem_group(track) for track in tracks}), 1)
        self.assertIsNotNone(stem_group(tracks[0]))
        self.press(0, 2)
        self.mix(25)
        frame = self.engine._groups[stem_group(tracks[0])].cursor
        self.press(3, 2)
        error = lag_samples(self.mix(20), frame, (1, 4))
        report("after save and load", error)
        self.assertLess(error, 0.25)

    def test_loading_a_project_stops_the_old_stems(self):
        self.split("Song")
        self.press(0, 0)
        path = os.path.join(self.root, "empty.soniclab")
        other = MainWindow()
        other.current_path = path
        other._save_project()
        other.close()
        other.deleteLater()
        self.window._load_project(path)
        self.assertEqual(self.engine.previews(), {})
        self.assertEqual(self.engine._groups, {})

    def test_stems_at_another_sample_rate_keep_their_length_and_sync(self):
        drums, bass, *_rest = self.split("Song", rate=48000)
        self.assertAlmostEqual(drums["duration"], SECONDS, delta=1.0 / OUTPUT_RATE)
        self.assertEqual(len(drums["_audio"]), len(bass["_audio"]))
        self.press(self.index(drums), 1)
        self.mix(30)
        frame = self.engine._groups[stem_group(drums)].cursor
        self.press(self.index(bass), 1)
        # The resampler scales the ramp a little (its DC gain), so compare
        # with the decoded audio itself: aligned, the mix is 3x drums.
        mixed = self.mix(20)
        expected = 3.0 * drums["_audio"][frame : frame + len(mixed), 0]
        error = float(np.max(np.abs(mixed - expected)) / STEP)
        report("48 kHz stems", error)
        self.assertLess(error, 0.25)

    def test_a_frame_with_eight_stems_stays_inside_60_fps(self):
        for group in (self.split("One"), self.split("Two")):
            for track in group:
                self.press(self.index(track), 0)
        out = np.zeros((BLOCK, 2), np.float32)
        costs = []
        for _ in range(60):
            self.engine._callback(out, BLOCK, None, None)
            started = time.perf_counter()
            self.window._animate_playheads()
            self.app.processEvents()
            costs.append(time.perf_counter() - started)
        median = float(np.median(costs))
        print(f"\n  UI frame with 8 stem voices: median {median * 1000:.2f} ms", end="")
        self.assertLess(median, 0.016)


if __name__ == "__main__":
    unittest.main()
