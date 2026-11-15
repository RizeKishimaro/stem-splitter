"""The dancer follows the playhead. It does not keep a clock of its own."""

import inspect
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.dancer import (
    FRAME_PX,
    SHEET_COLUMNS,
    Hit,
    HitDeck,
    detect_hits,
    fold_loop,
    layer_levels,
    load_blocks,
    phrase_beat,
    pose_at,
    sheet_frame,
    sustain_bass,
)
from soniclab.ui.dancer_view import DancerView, sheet_image

# 120 BPM: a bar is 2 seconds, a beat is 0.5 seconds.
DOWNS = tuple(float(bar) * 2.0 for bar in range(8))


class PhraseTests(unittest.TestCase):
    def test_a_downbeat_opens_the_phrase_and_the_next_one_opens_bar_two(self):
        self.assertAlmostEqual(phrase_beat(0.0, 120, DOWNS), 0.0)
        self.assertEqual(pose_at(0.0, 120, DOWNS).frame, 0)
        self.assertTrue(pose_at(0.0, 120, DOWNS).on_downbeat)
        # Bar 2 is beat 5 of the choreography: smooth frame 48.
        self.assertAlmostEqual(phrase_beat(2.0, 120, DOWNS), 4.0)
        self.assertEqual(pose_at(2.0, 120, DOWNS).frame, 48)
        self.assertTrue(pose_at(2.0, 120, DOWNS).on_downbeat)
        # The downbeat after that resets the phrase, it does not keep counting.
        self.assertAlmostEqual(phrase_beat(4.0, 120, DOWNS), 0.0)
        self.assertEqual(pose_at(4.0, 120, DOWNS).frame, 0)

    def test_the_authored_sheet_matches_the_choreography_keys(self):
        blocks = load_blocks()
        smooth = blocks["smooth"]["frames"]
        pixel = blocks["pixel"]["frames"]
        self.assertEqual(len(smooth), 96)
        self.assertEqual(len(pixel), 32)
        self.assertEqual(smooth[0]["beat"], 1)
        self.assertEqual(smooth[48]["beat"], 5)
        self.assertEqual(smooth[48]["step_in_beat"], 0)
        # Pixel mode holds the sixteenth. Those are every third smooth frame.
        for step in range(32):
            self.assertEqual(sheet_frame(step / 4.0, "pixel"), step * 3)

    def test_bpm_changes_move_the_grid_until_downbeats_take_over(self):
        at_120 = phrase_beat(1.0, 120, ())
        at_60 = phrase_beat(1.0, 60, ())
        self.assertAlmostEqual(at_120, 2.0)
        self.assertAlmostEqual(at_60, 1.0)
        self.assertNotEqual(pose_at(1.0, 120, ()).frame, pose_at(1.0, 60, ()).frame)
        # Detected bars are 1 second apart. Half a bar is beat 3, whatever the
        # project BPM field says.
        tight = (0.0, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(phrase_beat(0.5, 90, tight), 2.0)
        self.assertAlmostEqual(phrase_beat(0.5, 160, tight), 2.0)

    def test_seeking_jumps_the_pose_and_stays_there(self):
        early = pose_at(0.25, 120, DOWNS)
        later = pose_at(3.0, 120, DOWNS)
        self.assertNotEqual(early.frame, later.frame)
        self.assertEqual(pose_at(3.0, 120, DOWNS), later)
        time.sleep(0.05)
        self.assertEqual(pose_at(3.0, 120, DOWNS), later)

    def test_a_loop_wraps_back_to_the_same_pose(self):
        loop = (2.0, 6.0)
        self.assertAlmostEqual(fold_loop(6.0, loop), 2.0)
        self.assertEqual(pose_at(6.0, 120, DOWNS, loop), pose_at(2.0, 120, DOWNS, loop))
        self.assertEqual(pose_at(5.5, 120, DOWNS, loop).frame, pose_at(5.5, 120, DOWNS).frame)
        # One loop later is the same place in the phrase.
        self.assertEqual(pose_at(10.0, 120, DOWNS, loop), pose_at(6.0, 120, DOWNS, loop))

    def test_stop_holds_the_downbeat_pose(self):
        stopped = pose_at(0.0, 120, DOWNS)
        self.assertEqual(stopped.frame, 0)
        self.assertTrue(stopped.on_downbeat)
        self.assertEqual(stopped.drop_blocks, 0.0)


def _pulse(rate, hz, seconds):
    count = int(rate * seconds)
    times = np.arange(count, dtype=np.float32) / rate
    return np.sin(2 * np.pi * hz * times).astype(np.float32)


def _add(audio, signal, at, rate):
    start = int(at * rate)
    if start >= audio.size:
        return
    end = min(audio.size, start + signal.size)
    audio[start:end] += signal[: end - start]


def _mix(kind, rate, bpm=125.0, seconds=2.0):
    """Same tempo, different drums. ``beat`` is the clock with no drums."""
    audio = np.zeros(int(rate * seconds), dtype=np.float32)
    beat = 60.0 / bpm
    kick = _pulse(rate, 60, 0.018)
    hat = _pulse(rate, 9000, 0.008)
    noise = np.random.default_rng(1).uniform(-1, 1, int(rate * 0.02)).astype(np.float32)
    if kind in ("kick", "ks", "ksh", "dense"):
        for at in np.arange(0.0, seconds - 0.05, beat * 2):
            _add(audio, kick, at, rate)
    if kind in ("ks", "ksh", "dense"):
        for at in np.arange(beat, seconds - 0.05, beat * 2):
            _add(audio, noise, at, rate)
    if kind in ("ksh", "dense"):
        for at in np.arange(0.04, seconds - 0.05, beat / 2):
            _add(audio, hat * 0.8, at, rate)
    if kind == "dense":
        extra = np.random.default_rng(2).uniform(-1, 1, int(rate * 0.012)).astype(np.float32)
        for at in np.arange(0.03, seconds - 0.05, 0.045):
            _add(audio, extra * 0.7, at, rate)
    return audio


class LayerTests(unittest.TestCase):
    def test_the_same_bpm_dances_at_the_same_speed(self):
        bpm = 125.0
        times = [index * 0.02 for index in range(80)]
        plain = [pose_at(t, bpm).frame for t in times]
        mixes = {}
        for kind in ("beat", "kick", "ks", "ksh", "dense"):
            audio = _mix(kind, OUTPUT_RATE, bpm)
            hits = detect_hits(audio, OUTPUT_RATE, 0.0)
            mixes[kind] = hits
            frames = [pose_at(t, bpm, hits=hits).frame for t in times]
            self.assertEqual(frames, plain, kind)
        self.assertFalse(mixes["beat"])
        self.assertTrue(any(hit.kind == "kick" for hit in mixes["kick"]))
        self.assertFalse(any(hit.kind in ("snare", "hat") for hit in mixes["kick"]))
        self.assertTrue(any(hit.kind == "snare" for hit in mixes["ks"]))
        self.assertFalse(any(hit.kind == "hat" for hit in mixes["ks"]))
        self.assertTrue(any(hit.kind == "hat" for hit in mixes["ksh"]))
        self.assertGreater(len(mixes["dense"]), len(mixes["ksh"]))
        # A kick drops the body. Taking the hats away does not move the frame.
        kick = next(hit for hit in mixes["kick"] if hit.kind == "kick")
        dropped = pose_at(kick.time + 0.01, bpm, hits=mixes["kick"])
        quiet = pose_at(kick.time + 0.01, bpm)
        self.assertEqual(dropped.frame, quiet.frame)
        self.assertGreater(dropped.drop_blocks, 0.4)
        self.assertEqual(quiet.drop_blocks, 0.0)
        hat = next(hit for hit in mixes["ksh"] if hit.kind == "hat")
        with_hat = pose_at(hat.time + 0.004, bpm, hits=mixes["ksh"])
        without = pose_at(hat.time + 0.004, bpm, hits=mixes["ks"])
        self.assertEqual(with_hat.frame, without.frame)
        self.assertGreater(with_hat.layers.hat, 0.35)
        self.assertEqual(without.layers.hat, 0.0)

    def test_hits_accent_the_current_pose_and_do_not_advance_it(self):
        # Beat 0.15 is between sixteenths, so the smooth frame is an in-between.
        self.assertAlmostEqual(phrase_beat(0.075, 120, DOWNS), 0.15)
        # A few milliseconds later the accent has risen. The phrase has not.
        base = pose_at(0.079, 120, DOWNS)
        self.assertNotEqual(base.frame % 3, 0)
        snare = pose_at(0.079, 120, DOWNS, hits=[Hit("snare", 0.075, 1.0)])
        hat = pose_at(0.079, 120, DOWNS, hits=[Hit("hat", 0.075, 1.0)])
        self.assertEqual(snare.frame, base.frame)
        self.assertEqual(hat.frame, base.frame)
        self.assertEqual(snare.draw_frame % 3, 0)
        self.assertEqual(hat.draw_frame, base.frame)
        self.assertGreater(hat.hat_blocks, 0.0)

    def test_a_hit_is_drawn_when_it_is_heard_and_not_before(self):
        audio = np.zeros(OUTPUT_RATE, dtype=np.float32)
        _add(audio, _pulse(OUTPUT_RATE, 55, 0.016), 0.40, OUTPUT_RATE)
        hits = detect_hits(audio, OUTPUT_RATE, 0.0)
        kick = next(hit for hit in hits if hit.kind == "kick")
        stamp_ms = (kick.time - 0.40) * 1000.0
        self.assertGreaterEqual(stamp_ms, 0.0)
        self.assertLess(stamp_ms, 15.0)
        self.assertEqual(layer_levels(kick.time - 0.001, [kick]).kick, 0.0)
        when = kick.time
        while when < kick.time + 0.02 and layer_levels(when, [kick]).kick < 0.5:
            when += 0.0005
        lag_ms = (when - kick.time) * 1000.0
        print(f"\n  kick stamp {stamp_ms:.1f} ms, layer {lag_ms:.1f} ms", end="")
        self.assertGreaterEqual(lag_ms, 0.0)
        self.assertLess(lag_ms, 12.0)
        # The render head is 30 ms ahead of the speakers. The drop waits.
        latency = 0.030
        self.assertEqual(layer_levels(kick.time - latency, [kick]).kick, 0.0)
        self.assertGreater(layer_levels(kick.time + 0.004, [kick]).kick, 0.5)

    def test_a_held_bass_is_not_a_kick_and_a_snare_is_not_a_hat(self):
        held = np.sin(2 * np.pi * 80 * np.arange(OUTPUT_RATE // 2) / OUTPUT_RATE).astype(np.float32)
        self.assertEqual(detect_hits(held, OUTPUT_RATE), [])
        self.assertGreater(sustain_bass(held, OUTPUT_RATE), 0.35)
        swayed = pose_at(0.2, 125, bass=sustain_bass(held, OUTPUT_RATE))
        self.assertGreater(abs(swayed.sway_blocks), 0.0)
        self.assertEqual(swayed.drop_blocks, 0.0)
        times = np.arange(int(0.4 * OUTPUT_RATE)) / OUTPUT_RATE
        pluck = (np.sin(2 * np.pi * 80 * times) * np.exp(-times * 3)).astype(np.float32)
        bed = np.zeros(OUTPUT_RATE, dtype=np.float32)
        _add(bed, pluck, 0.2, OUTPUT_RATE)
        self.assertFalse(any(hit.kind == "kick" for hit in detect_hits(bed, OUTPUT_RATE)))
        noise = np.zeros(OUTPUT_RATE, dtype=np.float32)
        burst_n = np.random.default_rng(0).uniform(-1, 1, int(0.02 * OUTPUT_RATE)).astype(np.float32)
        _add(noise, burst_n, 0.3, OUTPUT_RATE)
        kinds = {hit.kind for hit in detect_hits(noise, OUTPUT_RATE)}
        self.assertIn("snare", kinds)
        self.assertNotIn("hat", kinds)
        bright = np.zeros(OUTPUT_RATE, dtype=np.float32)
        _add(bright, _pulse(OUTPUT_RATE, 9000, 0.01), 0.3, OUTPUT_RATE)
        hat_kinds = {hit.kind for hit in detect_hits(bright, OUTPUT_RATE)}
        self.assertIn("hat", hat_kinds)
        self.assertNotIn("kick", hat_kinds)
        # The live view only sees a few dozen milliseconds. A bass attack can
        # look like a kick until the note is still there a tenth later.
        streamed = np.zeros(int(0.8 * OUTPUT_RATE), dtype=np.float32)
        _add(streamed, pluck, 0.1, OUTPUT_RATE)
        deck = HitDeck()
        window, hop = 2048, 1024
        for start in range(0, streamed.size - window, hop):
            deck.update(streamed[start : start + window], OUTPUT_RATE, start / OUTPUT_RATE)
        self.assertFalse(any(hit.kind == "kick" for hit in deck.hits))
        short = np.zeros(int(0.8 * OUTPUT_RATE), dtype=np.float32)
        _add(short, _pulse(OUTPUT_RATE, 55, 0.016), 0.1, OUTPUT_RATE)
        deck = HitDeck()
        for start in range(0, short.size - window, hop):
            deck.update(short[start : start + window], OUTPUT_RATE, start / OUTPUT_RATE)
        self.assertTrue(any(hit.kind == "kick" for hit in deck.hits))

    def test_the_same_hit_is_not_triggered_twice(self):
        audio = np.zeros(OUTPUT_RATE, dtype=np.float32)
        _add(audio, _pulse(OUTPUT_RATE, 55, 0.016), 0.4, OUTPUT_RATE)
        deck = HitDeck()
        deck.update(audio, OUTPUT_RATE, 0.0)
        once = len(deck.hits)
        self.assertGreater(once, 0)
        deck.update(audio, OUTPUT_RATE, 0.0)
        self.assertEqual(len(deck.hits), once)

    def test_pause_holds_the_dance_and_a_seek_clears_hits(self):
        app = QApplication.instance() or QApplication([])
        state = {"seconds": 0.5, "playing": True}
        audio = np.zeros(int(0.3 * OUTPUT_RATE), dtype=np.float32)
        _add(audio, _pulse(OUTPUT_RATE, 55, 0.016), 0.1, OUTPUT_RATE)
        view = DancerView()
        view.set_clock(lambda: (state["seconds"], 125.0, (), None, state["playing"]))
        view.set_audio(lambda: (audio, 0.35))
        view._tick()
        self.assertGreater(view.pose.layers.kick, 0.3)
        held = view.pose.frame
        state["playing"] = False
        view._tick()
        self.assertEqual(view.pose.frame, held)
        self.assertGreater(view.pose.layers.kick, 0.3)
        state["seconds"] = 0.0
        state["playing"] = True
        view._tick()
        self.assertEqual(view._deck.hits, [])
        self.assertEqual(view.pose.layers.kick, 0.0)
        view._timer.stop()
        view.close()
        app.processEvents()


class PaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_painting_a_cached_frame_holds_60fps(self):
        image = sheet_image()
        self.assertIsNotNone(image)
        self.assertGreaterEqual(image.width(), SHEET_COLUMNS * FRAME_PX)
        view = DancerView()
        view.set_clock(lambda: (0.0, 120.0, DOWNS, None, False))
        view.resize(240, 260)
        view.show()
        self.app.processEvents()
        view._tick()
        started = time.perf_counter()
        frames = 120
        for _ in range(frames):
            picture = QImage(view.stage.size(), QImage.Format.Format_ARGB32_Premultiplied)
            painter = QPainter(picture)
            view.paint_stage(painter, picture.width(), picture.height())
            painter.end()
        elapsed = time.perf_counter() - started
        mean_ms = elapsed / frames * 1000.0
        print(f"\n  dancer paint mean {mean_ms:.3f} ms ({1000.0 / mean_ms:.1f} fps)", end="")
        self.assertLess(mean_ms, 16.6)
        view._timer.stop()
        view.close()

    def test_the_callback_still_fills_its_block_while_the_dancer_only_reads(self):
        engine = AudioEngine()
        engine._capture_enabled = True
        tone = np.full((OUTPUT_RATE, 2), 0.2, dtype=np.float32)
        engine.add_clip(tone, 0.0)
        engine._playing = True
        block = 512
        out = np.zeros((block, 2), np.float32)
        started = time.perf_counter()
        for _ in range(80):
            engine._callback(out, block, None, None)
        callback_ms = (time.perf_counter() - started) / 80 * 1000.0
        heard = engine.read_master(block)
        again = engine.read_master(block)
        self.assertEqual(heard.shape[0], block)
        self.assertTrue(np.array_equal(heard, again))  # reading does not consume the block
        self.assertGreater(float(np.max(np.abs(out))), 0.0)
        source = inspect.getsource(AudioEngine._callback)
        self.assertNotIn("detect_hits", source)
        self.assertNotIn("dancer", source)
        analysis_started = time.perf_counter()
        window = np.tile(heard[:, 0], 4)
        for _ in range(40):
            detect_hits(window, OUTPUT_RATE, 0.0)
        analysis_ms = (time.perf_counter() - analysis_started) / 40 * 1000.0
        budget_ms = block / OUTPUT_RATE * 1000.0
        print(
            f"\n  callback {callback_ms:.3f} ms, detect {analysis_ms:.3f} ms, block {budget_ms:.2f} ms",
            end="",
        )
        self.assertLess(callback_ms, budget_ms)
        self.assertLess(analysis_ms, budget_ms * 0.5)


if __name__ == "__main__":
    unittest.main()
