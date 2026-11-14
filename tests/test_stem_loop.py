"""Synced stem loops: picked stems of one split repeat one region of the
song on a shared clock that the audio callback wraps on the sample.

Stems play ramps (see test_stem_sync), so the frame each stem is on can be
read off the mix. Outside the short wrap crossfade the mix must equal the
ramps at the expected loop frame exactly.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import LIMIT_CEILING, LOOP_XFADE, OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import loop_group, stem_block
from soniclab.ui.timeline import stem_group, stem_loop
from tests.test_stem_sync import BLOCK, STEP, SharedClockWindowTests, ramp, report, run

LOOP = (1.0, 1.5)
L0, L1 = int(LOOP[0] * OUTPUT_RATE), int(LOOP[1] * OUTPUT_RATE)
CYCLE = L1 - L0  # 22050: not a whole number of blocks


def frames_from(first: int, count: int, loop=(L0, L1)) -> np.ndarray:
    """The song frame of each output sample, starting at ``first``."""
    raw = first + np.arange(count)
    return np.where(raw >= loop[1], loop[0] + (raw - loop[0]) % (loop[1] - loop[0]), raw)


def steady(frames: np.ndarray, loop=(L0, L1)) -> np.ndarray:
    """Samples outside the crossfade after each wrap."""
    return ~((frames >= loop[0]) & (frames < loop[0] + LOOP_XFADE))


def lag(mixed, frames, scales, regions=None, loop=(L0, L1)) -> float:
    """Worst lag in samples over the steady samples. ``regions`` gives each
    stem's (first, end) file frames; outside it the stem is silent."""
    expected = np.zeros(len(frames))
    for index, scale in enumerate(scales):
        inside = np.ones(len(frames), bool)
        if regions is not None and regions[index] is not None:
            inside = (frames >= regions[index][0]) & (frames < regions[index][1])
        expected += np.where(inside, frames * STEP * scale, 0.0)
    keep = steady(frames, loop)
    if regions is not None:
        for region in regions:
            if region is not None:  # the region's own edges are not lags
                for edge in region:
                    keep &= np.abs(frames - edge) > 1
    return float(np.max(np.abs(expected[keep] - mixed[keep])) / (STEP * min(scales)))


class LoopEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        for scale in (1, 2, 3, 4):
            self.engine.add_clip(ramp(scale), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def cursor(self, group="loop"):
        return self.engine._groups[group].cursor

    def test_every_cycle_wraps_on_the_sample_with_no_drift(self):
        self.engine.stem_play("loop", [("a", 0), ("b", 1)], 0.0, loop=LOOP)
        self.assertEqual(self.cursor(), L0)  # a loop starts at its start
        blocks = 6 * CYCLE // BLOCK
        mixed = run(self.engine, blocks)
        frames = frames_from(L0, len(mixed))
        error = lag(mixed, frames, (1, 2))
        report(f"loop over {len(mixed) / CYCLE:.1f} cycles", error)
        self.assertLess(error, 0.25)
        # Every cycle after the first (which starts unfaded) is identical,
        # sample for sample: the loop neither drifts nor gains a sample.
        for cycle in (2, 3, 4):
            np.testing.assert_array_equal(mixed[CYCLE : 2 * CYCLE], mixed[cycle * CYCLE : (cycle + 1) * CYCLE])
        self.assertEqual(self.cursor(), frames_from(L0, blocks * BLOCK + 1)[-1])

    def test_a_late_stem_joins_the_loop_where_it_is(self):
        self.engine.stem_play("loop", [("a", 0)], 0.0, loop=LOOP)
        run(self.engine, int(1.7 * CYCLE) // BLOCK)
        joined = self.cursor()
        self.assertGreater(joined, L0 + BLOCK)  # not the loop start
        self.engine.stem_play("loop", [("c", 2)], 0.0, loop=LOOP)
        mixed = run(self.engine, 3 * CYCLE // BLOCK)
        error = lag(mixed, frames_from(joined, len(mixed)), (1, 3))
        report("late join into a loop", error)
        self.assertLess(error, 0.25)

    def test_pause_and_resume_one_stem_keeps_the_others(self):
        self.engine.stem_play("loop", [("a", 0), ("b", 1)], 0.0, loop=LOOP)
        run(self.engine, 20)
        self.engine.preview_pause("b")
        frame = self.cursor()
        alone = run(self.engine, 50)
        self.assertLess(lag(alone, frames_from(frame, len(alone)), (1,)), 0.25)
        self.engine.stem_play("loop", [("b", 1)])
        frame = self.cursor()
        both = run(self.engine, 60)
        error = lag(both, frames_from(frame, len(both)), (1, 2))
        report("loop resume after pause", error)
        self.assertLess(error, 0.25)

    def test_source_offsets_and_places_are_kept(self):
        # b holds file 1.2..2.0 s and sits at 7 s in the arrangement.
        self.engine.set_clip_region(1, 7.0, 1.2, 0.8)
        self.engine.stem_play("loop", [("a", 0), ("b", 1)], 0.0, loop=LOOP)
        mixed = run(self.engine, 3 * CYCLE // BLOCK)
        b_region = (int(1.2 * OUTPUT_RATE), int(2.0 * OUTPUT_RATE))
        error = lag(mixed, frames_from(L0, len(mixed)), (1, 2), regions=(None, b_region))
        report("loop with an offset stem", error)
        self.assertLess(error, 0.25)

    def test_a_stem_shorter_than_the_loop_rests_and_comes_back(self):
        short = self.engine.add_clip(ramp(5, seconds=1.25), 0.0)
        self.engine.stem_play("loop", [("a", 0), ("s", short)], 0.0, loop=LOOP)
        mixed = run(self.engine, 4 * CYCLE // BLOCK)
        error = lag(mixed, frames_from(L0, len(mixed)), (1, 5), regions=(None, (0, int(1.25 * OUTPUT_RATE))))
        self.assertLess(error, 0.25)
        self.assertIn("s", self.engine.previews())  # silent past its end, still looping

    def test_moving_the_boundaries_while_it_plays(self):
        self.engine.stem_play("loop", [("a", 0), ("b", 1)], 0.0, loop=LOOP)
        run(self.engine, 30)
        # A later loop: the clock plays on to it, then wraps at its end.
        later = (2.0, 2.3)
        self.engine.group_set_loop("loop", later)
        frame = self.cursor()
        new = (round(later[0] * OUTPUT_RATE), round(later[1] * OUTPUT_RATE))  # as the engine rounds
        mixed = run(self.engine, 3 * (new[1] - L0) // BLOCK)
        self.assertLess(lag(mixed, frames_from(frame, len(mixed), new), (1, 2), loop=new), 0.25)
        self.assertTrue(new[0] <= self.cursor() < new[1])
        # An end before the clock: it wraps to the start on the next block.
        self.engine.group_set_loop("loop", (2.0, 2.1))
        mixed = run(self.engine, 1)
        self.assertTrue(round(2.0 * OUTPUT_RATE) <= self.cursor() < round(2.1 * OUTPUT_RATE))
        steps = np.abs(np.diff(mixed))
        self.assertLess(float(steps.max()), 3 * 0.3 * OUTPUT_RATE * STEP / LOOP_XFADE * 1.5)

    def test_the_wrap_has_no_click(self):
        self.engine.stem_play("loop", [("a", 0), ("b", 1)], 0.0, loop=LOOP)
        mixed = run(self.engine, 3 * CYCLE // BLOCK)
        jump = CYCLE * STEP * 3  # what an unfaded wrap would step by
        biggest = float(np.max(np.abs(np.diff(mixed))))
        print(f"\n  largest step at a wrap {biggest:.2e} vs {jump:.2e} unfaded", end="")
        self.assertLess(biggest, jump / LOOP_XFADE * 1.5)

    def test_previews_stay_inside_the_loop(self):
        self.engine.stem_play("loop", [("a", 0)], 0.0, loop=LOOP)
        for _ in range(3 * CYCLE // BLOCK):
            run(self.engine, 1)
            seconds = self.engine.previews(now=time.monotonic() + 0.004)["a"][1]
            self.assertTrue(LOOP[0] <= seconds < LOOP[1], seconds)

    def test_only_the_loops_stems_loop(self):
        self.engine.stem_play("split/loop", [("a", 0)], 0.0, loop=LOOP)
        self.engine.stem_play("split", [("d", 3)], 3.0)
        run(self.engine, 2 * CYCLE // BLOCK)
        self.assertTrue(L0 <= self.cursor("split/loop") < L1)
        self.assertEqual(self.cursor("split"), 3 * OUTPUT_RATE + (2 * CYCLE // BLOCK) * BLOCK)

    def test_loud_loops_never_clip(self):
        engine = AudioEngine()
        engine._capture_enabled = False
        for _ in range(4):
            engine.add_clip(np.full((OUTPUT_RATE * 2, 2), 0.9, np.float32), 0.0)
        with patch.object(engine, "_ensure_stream", lambda: None):
            engine.stem_play("loop", [(k, k) for k in range(4)], 0.0, loop=(0.2, 0.3))
            peaks = [float(np.max(np.abs(run(engine, 1)))) for _ in range(60)]
        self.assertLessEqual(max(peaks), LIMIT_CEILING + 1e-6)

    def test_short_loops_stay_well_inside_the_deadline(self):
        for scale in (5, 6, 7, 8):
            self.engine.add_clip(ramp(scale), 0.0)
        self.engine.stem_play("one", [(k, k) for k in range(4)], 0.0, loop=(1.0, 1.05))
        self.engine.stem_play("two", [(k, k) for k in range(4, 8)], 0.0, loop=(2.0, 2.06))
        self.engine._playing = True
        out = np.zeros((BLOCK, 2), np.float32)
        costs = []
        for _ in range(400):
            started = time.perf_counter()
            self.engine._callback(out, BLOCK, None, None)
            costs.append(time.perf_counter() - started)
        self.engine._playing = False
        p99 = float(np.percentile(costs, 99))
        print(f"\n  looping callback p99 {p99 * 1000:.3f} ms of {BLOCK / OUTPUT_RATE * 1000:.2f} ms", end="")
        self.assertLess(p99, BLOCK / OUTPUT_RATE / 3)


class LoopWindowTests(SharedClockWindowTests):
    """Picking stems, ▶ on any member, markers, undo and save/load."""

    def loop_two(self):
        drums, bass, other, vocals = self.split("Song")
        view = self.window.timeline
        view.set_picks({self.index(drums): {1}, self.index(bass): {2}})
        self.window._loop_picked_stems()
        return drums, bass, other, vocals

    def test_picked_stems_become_one_synced_loop(self):
        drums, bass, other, vocals = self.loop_two()
        self.assertEqual(stem_loop(drums), (2.0, 6.0))
        self.assertEqual(stem_loop(bass), (2.0, 6.0))
        self.assertIsNone(stem_loop(other))
        clip = self.window.timeline._clip_for(self.index(bass))
        self.assertIsNotNone(clip._stem_loop)  # its markers are drawn
        self.assertIsNone(self.window.timeline._clip_for(self.index(vocals))._stem_loop)

    def test_any_member_starts_it_and_the_others_join(self):
        drums, bass, other, vocals = self.loop_two()
        scroll = self.window.timeline.horizontalScrollBar().value()
        self.press(self.index(bass), 0)  # bar 1 is outside the loop: it starts at 2.0
        group = loop_group(stem_group(drums))
        self.assertEqual(self.engine._groups[group].cursor, 2 * OUTPUT_RATE)
        self.mix(300)
        joined = self.engine._groups[group].cursor
        self.press(self.index(drums), 3)
        self.press(self.index(vocals), 0)  # not in the loop: the split's own clock
        self.assertIn(stem_group(drums), self.engine._groups)  # vocals: a clock of its own
        self.engine.preview_pause((self.index(vocals), stem_block(0)))
        loop = (2 * OUTPUT_RATE, 6 * OUTPUT_RATE)
        mixed = self.mix(450)  # past the loop end and around again
        error = lag(mixed, frames_from(joined, len(mixed), loop), (1, 2), loop=loop)
        report("window: loop join and wrap", error)
        self.assertLess(error, 0.25)
        self.assertFalse(self.engine.playing)
        self.assertEqual(self.engine.position, 0.0)
        self.assertEqual(self.window.timeline.horizontalScrollBar().value(), scroll)

    def test_dragging_a_marker_moves_the_loop_for_every_member(self):
        drums, bass, *_rest = self.loop_two()
        self.press(self.index(drums), 1)
        self.mix(5)
        clip = self.window.timeline._clip_for(self.index(bass))
        clip.stem_loop_callback(clip, 1, 4.5, False)  # mid-drag: heard at once
        group = loop_group(stem_group(drums))
        self.assertEqual(self.engine._groups[group].loop, (2 * OUTPUT_RATE, int(4.5 * OUTPUT_RATE)))
        clip.stem_loop_callback(clip, 1, 5.0, True)
        self.assertEqual(stem_loop(drums), (2.0, 5.0))
        self.assertEqual(stem_loop(bass), (2.0, 5.0))
        self.assertAlmostEqual(clip._stem_loop[1], 5.0 * clip.pixels_per_second, places=1)
        clip.stem_loop_callback(clip, 0, 4.99, True)  # never shorter than the minimum
        self.assertAlmostEqual(stem_loop(drums)[1] - stem_loop(drums)[0], 0.05, places=6)
        self.window._undo()
        self.window._undo()
        self.assertEqual(stem_loop(self.window.project.tracks[0]), (2.0, 6.0))  # undo made new track dicts

    def test_save_and_load_keep_the_loop(self):
        self.loop_two()
        path = os.path.join(self.root, "loop.soniclab")
        self.window.current_path = path
        self.window._save_project()
        self.window._load_project(path)
        tracks = self.window.project.tracks
        self.assertEqual([stem_loop(track) for track in tracks], [(2.0, 6.0), (2.0, 6.0), None, None])
        self.press(1, 0)
        self.assertEqual(self.engine._groups[loop_group(stem_group(tracks[0]))].cursor, 2 * OUTPUT_RATE)

    def test_clearing_the_loop_stops_its_stems(self):
        drums, *_rest = self.loop_two()
        self.press(self.index(drums), 1)
        self.window._clear_stem_loop(self.index(drums))
        self.assertIsNone(stem_loop(drums))
        self.assertEqual(self.engine.previews(), {})


# The inherited shared-clock tests already run from test_stem_sync.
for _name in [name for name in vars(SharedClockWindowTests) if name.startswith("test_")]:
    setattr(LoopWindowTests, _name, None)
del SharedClockWindowTests


if __name__ == "__main__":
    unittest.main()
