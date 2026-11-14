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
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
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
    def test_picked_bars_side_by_side_play_as_one_block(self):
        drums, bass, other, vocals = self.split("Song")
        view = self.window._view_for(self.index(other))
        # Vocals bars 2 and 3 side by side, other bar 2 alone, bass bar 4.
        view.set_picks({self.index(other): {1}, self.index(vocals): {1, 2}, self.index(bass): {3}})
        self.press(self.index(vocals), 2)  # ▶ on a picked block: the whole pick
        group = stem_group(drums)
        self.assertEqual(self.engine._groups[group].cursor, 2 * OUTPUT_RATE)
        self.assertEqual(self.engine._groups[group].loop, (2 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        run = (self.index(vocals), stem_block(1))
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertEqual(playing, {(self.index(other), stem_block(1)), run, (self.index(bass), stem_block(3))})
        cycle = 2 * OUTPUT_RATE
        mixed = self.mix(2 * 4 * OUTPUT_RATE // BLOCK + 1)  # twice round the 4 s loop
        fade = 64
        # First 2 s: other bar 2, vocals bar 2, bass bar 4 (one bar, then silent).
        first = lag_samples(mixed[: cycle - fade], cycle, (3, 4, 2), (0, 0, 2 * cycle))
        # Next 2 s: vocals goes on into bar 3; the one-bar blocks loop their
        # own bar again, never the bar after it.
        second = lag_samples(mixed[cycle + fade : 2 * cycle - fade], 2 * cycle + fade, (4, 3, 2), (0, -cycle, cycle))
        report("window: one combined block and two one-bar loops", max(first, second))
        self.assertLess(max(first, second), 0.25)
        # The one-bar blocks wrap with a crossfade, not a click.
        steps = np.abs(np.diff(mixed[cycle - 2 * fade : cycle + 2 * fade]))
        print(f"\n  largest step where two blocks wrap: {steps.max():.2e}", end="")
        self.assertLess(steps.max(), 0.002)
        # Each line stays inside its own block, looping there.
        self.window._sync_previews()
        for track, bars in ((other, {1}), (bass, {3}), (vocals, {1, 2})):
            clip = view._clip_for(self.index(track))
            self.assertEqual(set(clip._preview), bars)
            span = next(sp for sp in view._spans_of(clip) if sp[0] == min(bars))
            last = next(sp for sp in view._spans_of(clip) if sp[0] == max(bars))
            for _playing, x in clip._preview.values():
                self.assertTrue(span[1] - 1 <= x <= last[2] + 1, (track["name"], x, span, last))
        loop = mixed[: 2 * cycle]
        self.assertTrue(np.array_equal(mixed[2 * cycle + fade : 4 * cycle - fade], loop[fade : 2 * cycle - fade]))
        # The run shows on both its bars, the ⏸ on each.
        clip = view._clip_for(self.index(vocals))
        self.window._sync_previews()
        self.assertEqual(set(clip._preview), {1, 2})
        # ▶ on any picked block again pauses the whole pick.
        self.press(self.index(other), 1)
        self.assertFalse(any(state[0] for state in self.engine.previews().values()))

    def test_pieces_of_one_stem_side_by_side_play_as_one(self):
        drums, bass, *_rest = self.split("Song")
        view = self.window._view_for(self.index(bass))
        view.set_picks({self.index(bass): {1, 2}})
        self.window._move_selected_to_other_canvas()  # cut into two clips
        self.app.processEvents()
        tracks = self.window.project.tracks
        pieces = sorted((i for i, t in enumerate(tracks) if t.get("canvas") == 1), key=lambda i: tracks[i]["start"])
        self.assertEqual(len(pieces), 2)
        other = self.window.timeline2
        other.set_picks({})
        bars = [other.block_at_time(i, tracks[i]["start"] + 0.5) for i in pieces]
        self.press(pieces[1], bars[1])  # the second piece first, then its neighbour
        self.mix(5)
        self.press(pieces[0], bars[0])
        group = stem_group(drums)
        self.assertEqual(self.engine._groups[group].loop, (2 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        self.assertEqual(self.engine._groups[group].cursor, 2 * OUTPUT_RATE)
        mixed = self.mix(4 * OUTPUT_RATE // BLOCK)
        error = lag_samples(mixed[: 4 * OUTPUT_RATE - 64], 2 * OUTPUT_RATE, (2,))
        report("window: two pieces of one stem as one block", error)
        self.assertLess(error, 0.25)  # bar 2 then bar 3, never both at once
        # The last piece ends its clip: the wrap fades it out, no click.
        around = self.mix(4)
        steps = np.abs(np.diff(np.concatenate((mixed[-256:], around))))
        print(f"\n  largest step where cut pieces wrap: {steps.max():.2e}", end="")
        self.assertLess(steps.max(), 0.002)
        # Both pieces show as playing.
        self.assertTrue(all(state[0] for state in self.engine.previews().values()))
        self.assertEqual(len(self.engine.previews()), 2)

    def click(self, track, bar=None, modifiers=Qt.KeyboardModifier.NoModifier):
        """A real mouse click: on that bar's ▶ of the track, or (no bar) on
        the clip away from its buttons."""
        view = self.window._view_for(track)
        clip = view._clip_for(track)
        from soniclab.ui.timeline import BarPlayButton
        if bar is None:
            rect = clip.sceneBoundingRect()
            spot = rect.topLeft() + type(rect.topLeft())(rect.width() * 0.07, rect.height() * 0.3)
        else:
            button = next(c for c in clip.childItems() if isinstance(c, BarPlayButton) and c.index == bar)
            view.centerOn(button)
            self.app.processEvents()
            spot = button.sceneBoundingRect().center()
        view.ensureVisible(clip.sceneBoundingRect().adjusted(0, 0, -clip.sceneBoundingRect().width() + 40, 0))
        self.app.processEvents()
        point = view.mapFromScene(spot)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, modifiers, point)
        self.app.processEvents()

    def test_shift_click_tracks_then_a_play_button_loops_that_bar_on_all(self):
        drums, bass, other, vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        self.click(self.index(drums))
        self.click(self.index(other), modifiers=Qt.KeyboardModifier.ShiftModifier)
        self.click(self.index(vocals), modifiers=Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(view.selected_indices(), sorted(self.index(t) for t in (drums, other, vocals)))
        # ▶ on a track that is not selected plays only that track.
        self.click(self.index(bass), 2)
        self.assertEqual(set(self.engine.previews()), {(self.index(bass), stem_block(2))})
        self.engine.preview_stop_all()
        # ▶ on a selected track plays that bar on every selected track at once.
        self.click(self.index(other), 2)
        self.assertEqual(view.selected_indices(), sorted(self.index(t) for t in (drums, other, vocals)))
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertEqual(playing, {(self.index(t), stem_block(2)) for t in (drums, other, vocals)})
        group = stem_group(drums)
        self.assertEqual(self.engine._groups[group].loop, (4 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        mixed = self.mix(20)
        error = lag_samples(mixed, 4 * OUTPUT_RATE, (1, 3, 4))
        report("window: three Shift+clicked tracks, one bar", error)
        self.assertLess(error, 0.25)
        # Shift+click again takes a track out of the selection.
        self.click(self.index(vocals), modifiers=Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(view.selected_indices(), sorted(self.index(t) for t in (drums, other)))
        # A right-click on a selected track takes it out too (no menu).
        self.click(self.index(vocals), modifiers=Qt.KeyboardModifier.ShiftModifier)
        clip = view._clip_for(self.index(vocals))
        point = view.mapFromScene(clip.sceneBoundingRect().center())
        self.window._clip_context_menu(point, view)
        self.assertEqual(view.selected_indices(), sorted(self.index(t) for t in (drums, other)))
        # Shift+▶ again on a selected track pauses the selected ones.
        self.click(self.index(drums), 2, Qt.KeyboardModifier.ShiftModifier)
        states = self.engine.previews()
        self.assertFalse(states[(self.index(drums), stem_block(2))][0])
        self.assertFalse(states[(self.index(other), stem_block(2))][0])
        self.assertTrue(states[(self.index(vocals), stem_block(2))][0])  # no longer selected

    def click_block(self, track, bar, modifiers=Qt.KeyboardModifier.NoModifier):
        """A real click on a block of a track, away from its ▶."""
        view = self.window._view_for(track)
        clip = view._clip_for(track)
        span = next(sp for sp in view._spans_of(clip) if sp[0] == bar)
        spot = clip.mapToScene(span[1] + (span[2] - span[1]) * 0.2, clip.rect().height() * 0.3)
        view.centerOn(spot)
        self.app.processEvents()
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, modifiers, view.mapFromScene(spot))
        self.app.processEvents()

    def test_ctrl_click_picks_single_blocks_that_play_together_and_escape_clears(self):
        drums, bass, other, vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        ctrl = Qt.KeyboardModifier.ControlModifier
        self.click(self.index(other), 3)  # something already looping
        self.click_block(self.index(drums), 1, ctrl)
        self.click(self.index(bass), 3, ctrl)  # Ctrl on the ▶ itself: picks, plays nothing
        self.assertEqual(set(self.engine.previews()), {(self.index(other), stem_block(3))})
        self.click_block(self.index(vocals), 1, ctrl)
        self.click_block(self.index(vocals), 2, ctrl)
        # One block each, not whole clips.
        self.assertEqual(view.picks(), {self.index(drums): {1}, self.index(bass): {3}, self.index(vocals): {1, 2}})
        self.assertEqual(view.selected_indices(), [self.index(other)])  # from its ▶; Ctrl added no clips
        self.click_block(self.index(bass), 3, ctrl)  # Ctrl+click again: unpicked
        self.assertEqual(view.picks(), {self.index(drums): {1}, self.index(vocals): {1, 2}})
        self.click(self.index(drums), 1)  # ▶ on one picked block plays them all, and only them
        self.assertEqual(set(self.engine.previews()), {(self.index(drums), stem_block(1)), (self.index(vocals), stem_block(1))})
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertEqual(self.engine._groups[stem_group(drums)].loop, (2 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        # Esc stops every block (paused ones too) and deselects the picked
        # blocks and any selected clips.
        self.click(self.index(other))
        view.set_picks({self.index(other): {0}})
        self.click(self.index(bass), 3)
        self.click(self.index(bass), 3)  # paused
        paused = (self.index(bass), stem_block(3))
        self.assertFalse(self.engine.previews()[paused][0])
        view.setFocus()
        QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertEqual(view.selected_indices(), [])
        self.assertEqual(view.picks(), {})
        states = self.engine.previews()
        self.assertNotIn(paused, states)
        self.assertNotIn(3, view._clip_for(self.index(bass))._preview)
        self.assertEqual(states, {})  # everything stopped
        self.assertFalse(self.engine.playing)

    def test_a_shift_drag_box_picks_blocks_that_play_and_move_as_one(self):
        drums, bass, other, _vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        view.horizontalScrollBar().setValue(0)
        view.verticalScrollBar().setValue(0)
        self.app.processEvents()
        first = view._clip_for(self.index(drums)).sceneBoundingRect()
        last = view._clip_for(self.index(bass)).sceneBoundingRect()
        pps = view.pixels_per_second
        # From the empty row below the last one, inside bar 2, up to bass
        # inside bar 3: the box covers vocals, other and bass, not drums.
        rows = [view._clip_for(i).sceneBoundingRect() for i in range(len(self.window.project.tracks))]
        bottom = max(r.bottom() for r in rows) + 20
        x0 = first.left() + 2.3 * pps
        x1 = first.left() + 5.7 * pps
        start = view.mapFromScene(x0, bottom)
        end = view.mapFromScene(x1, last.center().y())
        shift = Qt.KeyboardModifier.ShiftModifier
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, shift, start)
        QTest.mouseMove(view.viewport(), view.mapFromScene((x0 + x1) / 2, bottom - 30))
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, shift, end)
        self.app.processEvents()
        picks = view.picks()
        self.assertEqual(picks.get(self.index(bass)), {1, 2})
        self.assertNotIn(self.index(drums), picks)  # the box stopped at bass
        self.assertEqual(view.selected_indices(), [])  # bars, not whole tracks
        self.click(self.index(bass), 1)
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertTrue(all(key[0] != self.index(drums) for key in playing))
        self.assertEqual(self.engine._groups[stem_group(drums)].loop, (2 * OUTPUT_RATE, 6 * OUTPUT_RATE))

    def test_a_click_on_play_in_a_drag_box_plays_the_box(self):
        drums, bass, other, _vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        view.set_picks({self.index(drums): {1, 2}, self.index(bass): {1}})
        self.click(self.index(bass), 1)
        playing = {key for key, state in self.engine.previews().items() if state[0]}
        self.assertEqual(playing, {(self.index(drums), stem_block(1)), (self.index(bass), stem_block(1))})
        self.assertEqual(self.engine._groups[stem_group(drums)].loop, (2 * OUTPUT_RATE, 6 * OUTPUT_RATE))
        self.assertEqual(view.picks(), {self.index(drums): {1, 2}, self.index(bass): {1}})  # still picked
        # A block outside the box plays only itself, joining in step.
        self.click(self.index(other), 3)
        self.assertTrue(self.engine.previews()[(self.index(other), stem_block(3))][0])

    def test_picked_blocks_and_selected_clips_move_to_the_other_canvas(self):
        drums, bass, other, vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        view.set_picks({self.index(drums): {1}, self.index(bass): {1, 2}})
        self.window._move_selected_to_other_canvas()
        self.app.processEvents()
        moved = [t for t in self.window.project.tracks if t.get("canvas") == 1]
        self.assertEqual(len(moved), 3)  # drums bar 2, bass bars 2 and 3
        self.assertTrue(all(stem_group(t) == stem_group(drums) for t in moved))
        self.assertEqual(sorted(round(t["start"], 3) for t in moved), [2.0, 2.0, 4.0])
        # Moved pieces still play in step with the stems left behind.
        index = next(i for i, t in enumerate(self.window.project.tracks) if t.get("canvas") == 1)
        self.window.timeline2.set_picks({})
        self.press(index, 0)
        self.press(self.index(other), 1)
        frame = self.engine._groups[stem_group(drums)].cursor
        self.assertEqual(frame, 2 * OUTPUT_RATE)
        # Selected whole clips go together too.
        self.window.timeline.scene_obj.clearSelection()
        for clip in self.window.timeline._clip_items():
            if clip.clip_index in (self.index(other), self.index(vocals)):
                clip.setSelected(True)
        self.window._move_selected_to_other_canvas()
        self.app.processEvents()
        self.assertEqual(other.get("canvas"), 1)
        self.assertEqual(vocals.get("canvas"), 1)


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
        fade = 64  # the wrap's crossfade (bass bar 4 ends the file: it fades out)
        error = lag_samples(mixed[: wrap - fade], joined_at, (1, 2, 3), shifts)
        again = lag_samples(mixed[wrap + fade : wrap + cycle - fade], cycle + fade, (1, 2, 3), shifts)
        report("window: three stems on three bars", max(error, again))
        self.assertLess(max(error, again), 0.25)
        # Each pause button shows on the stem's own bar.
        states = self.engine.previews()
        view = self.window._view_for(self.index(bass))
        for track, bar in ((drums, 1), (bass, 3), (other, 0)):
            index = self.index(track)
            self.assertEqual(view.block_at_time(index, states[(index, stem_block(bar))][1]), bar)
        # Another bar of a playing stem (not beside it) loops too, in step.
        self.press(self.index(bass), 1)
        frame = self.engine._groups[group].cursor
        both = (self.index(bass), stem_block(3)), (self.index(bass), stem_block(1))
        self.assertTrue(all(self.engine.previews()[key][0] for key in both))
        self.assertEqual(self.engine._voices[both[1]].shift, 0)
        mixed = self.mix(10)
        error = lag_samples(mixed, frame, (1, 2, 2, 3), (0, 2 * cycle, 0, -cycle))
        report("window: one stem on two bars", error)
        self.assertLess(error, 0.25)
        # Pausing one bar leaves the other playing, and its line stops.
        self.press(self.index(bass), 1)
        paused = self.engine.previews()[both[1]]
        self.assertFalse(paused[0])
        self.assertTrue(self.engine.previews()[both[0]][0])
        self.mix(20)
        self.assertEqual(self.engine.previews(now=time.monotonic() + 0.5)[both[1]], paused)
        self.assertEqual(self.window._view_for(self.index(bass)).block_at_time(self.index(bass), paused[1]), 1)

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
