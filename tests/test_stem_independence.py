"""Independent stem voices, measured in the audio output.

Each stem is a sine at its own frequency, so the mixed output shows which
stems sound in every 50 ms window. Qt timers run between audio blocks, so a
stale scheduled callback would show up as a stem that comes back by itself.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_stem_independence
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from soniclab.core.audio_engine import COMPLETED, OUTPUT_RATE, PAUSED, PLAYING, STOPPED, AudioEngine
from soniclab.ui.main_window import stem_block
from tests.test_stem_sync import SharedClockWindowTests, ramp, run

FREQS = {"drums": 300.0, "bass": 500.0, "other": 700.0, "vocals": 900.0}
ORDER = tuple(FREQS)
WINDOW = int(0.05 * OUTPUT_RATE)
BLOCK = 512
HEARD = 50.0  # a sine of 0.2 reads about 110 in its bin; silence reads ~0


class VoiceStateEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        for scale in (1, 2):
            self.engine.add_clip(ramp(scale), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_states_follow_each_voice_on_its_own(self):
        self.engine.preview_play([("a", 0, 0.0, 0.5), ("b", 1, 0.0, 6.0)])
        self.assertEqual(self.engine.voice_state("a")[0], PLAYING)
        self.engine.preview_pause("b")
        self.assertEqual(self.engine.voice_state("b")[0], PAUSED)
        self.assertEqual(self.engine.voice_state("a")[0], PLAYING)
        run(self.engine, int(1.0 * OUTPUT_RATE / BLOCK))  # a ends by itself
        self.assertEqual(self.engine.voice_state("a")[0], COMPLETED)
        self.assertEqual(self.engine.voice_state("b")[0], PAUSED)  # still waiting for ▶
        self.engine.preview_stop("b")
        self.assertEqual(self.engine.voice_state("b")[0], STOPPED)

    def test_a_request_for_an_old_voice_is_ignored(self):
        self.engine.preview_play([("a", 0, 0.0, 6.0)])
        old = self.engine.voice_state("a")[1]
        self.engine.preview_stop("a")
        self.engine.preview_play([("a", 0, 0.0, 6.0)])  # same key, new voice
        new = self.engine.voice_state("a")[1]
        self.assertNotEqual(old, new)
        self.engine.preview_pause("a", old)  # late: meant for the stopped one
        self.engine.preview_stop("a", old)
        self.assertEqual(self.engine.voice_state("a"), (PLAYING, new))
        self.engine.preview_pause("a", new)
        self.engine.preview_resume(["a"], {"a": old})
        self.assertEqual(self.engine.voice_state("a"), (PAUSED, new))


class StemIndependenceTests(SharedClockWindowTests):
    def make(self, name):
        """Four stems of a split, each a sine at its own frequency."""
        seconds = np.arange(OUTPUT_RATE * 8) / OUTPUT_RATE
        stems = {}
        for stem, freq in FREQS.items():
            path = os.path.join(self.root, f"{name}-{stem}.wav")
            tone = (0.2 * np.sin(2 * np.pi * freq * seconds)).astype(np.float32)
            sf.write(path, np.column_stack((tone, tone)), OUTPUT_RATE, subtype="FLOAT")
            stems[stem] = path
        added = self.window._add_stem_tracks(name, stems)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        return added

    def listen(self, seconds) -> dict:
        """``stem -> [heard in each 50 ms window]`` of the mixed output."""
        out = np.zeros((BLOCK, 2), np.float32)
        chunks = []
        for n in range(int(seconds * OUTPUT_RATE / BLOCK)):
            self.engine._callback(out, BLOCK, None, None)
            chunks.append(out[:, 0].copy())
            if n % 3 == 0:
                self.app.processEvents()  # timers and posted callbacks run
                self.window._animate_playheads()
        mixed = np.concatenate(chunks)
        bins = np.fft.rfftfreq(WINDOW, 1 / OUTPUT_RATE)
        heard = {stem: [] for stem in FREQS}
        for start in range(0, len(mixed) - WINDOW + 1, WINDOW):
            spectrum = np.abs(np.fft.rfft(mixed[start : start + WINDOW] * np.hanning(WINDOW)))
            for stem, freq in FREQS.items():
                heard[stem].append(float(spectrum[np.argmin(np.abs(bins - freq))]))
        return heard

    def assertSounds(self, heard, stem, msg=""):
        self.assertTrue(all(level > HEARD for level in heard[stem]), f"{stem} dropped out {msg}: {np.round(heard[stem], 0)}")

    def assertSilent(self, heard, stem, msg=""):
        self.assertTrue(all(level < HEARD for level in heard[stem]), f"{stem} sounded {msg}: {np.round(heard[stem], 0)}")

    def check_pause(self, a, a_bar, others, wait=4.5):
        """Pause stem ``a``: the others play on at once and without a gap,
        ``a`` stays silent for longer than its block, then resumes without
        interrupting them. ``others`` are stem names."""
        a_name = self.window.project.tracks[a]["name"].split()[-1]
        self.window._toggle_block(a, a_bar)  # ⏸
        heard = self.listen(wait)  # longer than the paused block's own time
        self.assertSilent(heard, a_name, "while paused")
        for name in others:
            self.assertSounds(heard, name, "while another stem was paused")
        key = self.window._block_voice(a, a_bar, self.engine.previews())
        self.assertEqual(self.engine.voice_state(key)[0], PAUSED)  # no stale timer restarted it
        self.window._toggle_block(a, a_bar)  # ▶
        heard = self.listen(1.0)
        for name in (a_name, *others):
            self.assertSounds(heard, name, "after the resume")

    def test_pausing_one_of_two_stems_leaves_the_other_playing(self):
        drums, bass, *_rest = self.make("Song")
        self.press(self.index(drums), 0)
        self.press(self.index(bass), 2)
        heard = self.listen(0.5)
        self.assertSounds(heard, "drums")
        self.assertSounds(heard, "bass")
        self.check_pause(self.index(drums), 0, ("bass",))

    def test_three_and_four_stems_pause_one_at_a_time(self):
        tracks = self.make("Song")
        for bar, track in enumerate(tracks):
            self.press(self.index(track), bar)
        self.listen(0.3)
        for track in tracks[:3]:
            others = [name for name in ORDER if name != track["name"].split()[-1]]
            index = self.index(track)
            self.check_pause(index, tracks.index(track), others)

    def test_cached_and_fresh_stems_pause_alike(self):
        fresh = self.make("Song")
        stems = {name: track["path"] for name, track in zip(ORDER, fresh)}
        self.window._finish_split_stems("Song", stems, ORDER, None)  # the same files again
        self.app.processEvents()
        cached = self.window.project.tracks[len(fresh):]
        self.press(self.index(fresh[0]), 0)  # drums, fresh
        self.press(self.index(cached[1]), 1)  # bass, cached
        self.listen(0.3)
        self.check_pause(self.index(cached[1]), 1, ("drums",))
        self.check_pause(self.index(fresh[0]), 0, ("bass",))

    def test_a_row_of_a_loop_pauses_alone(self):
        drums, bass, other, _vocals = self.make("Song")
        # Three loops on one song clock, each started with its own ▶ (a
        # deselected selection would stop instead).
        self.press(self.index(other), 0)
        self.press(self.index(drums), 0)
        self.press(self.index(bass), 1)
        self.listen(0.5)
        self.window._toggle_block(self.index(other), 0)  # ⏸ other's loop alone
        heard = self.listen(8.0)  # well past its 2 s block's end
        self.assertSilent(heard, "other")
        self.assertSounds(heard, "drums")
        self.assertSounds(heard, "bass")
        self.window._toggle_block(self.index(other), 0)  # ▶ again
        heard = self.listen(1.0)
        for name in ("drums", "bass", "other"):
            self.assertSounds(heard, name)

    def test_stems_moved_to_the_other_canvas_pause_alone(self):
        drums, bass, *_rest = self.make("Song")
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(self.index(bass), 1, start=3.0)
        self.app.processEvents()
        self.press(self.index(drums), 0)
        bass_index = self.index(bass)
        bar = self.window._view_for(bass_index).block_at_time(bass_index, 3.5)
        self.press(bass_index, bar)
        self.listen(0.3)
        self.check_pause(self.index(drums), 0, ("bass",))
        self.check_pause(bass_index, bar, ("drums",))

    def test_volume_mute_and_stop_touch_one_stem(self):
        drums, bass, other, _vocals = self.make("Song")
        for bar, track in enumerate((drums, bass, other)):
            self.press(self.index(track), bar)
        before = self.listen(0.5)
        self.window._set_gain(self.index(bass), 0.5)
        after = self.listen(0.5)
        self.assertAlmostEqual(np.median(after["bass"]) / np.median(before["bass"]), 0.5, delta=0.05)
        self.assertAlmostEqual(np.median(after["drums"]) / np.median(before["drums"]), 1.0, delta=0.05)
        self.window._set_muted(self.index(bass), True)
        heard = self.listen(0.5)
        self.assertSilent(heard, "bass")
        self.assertSounds(heard, "drums")
        self.assertSounds(heard, "other")
        self.window._set_muted(self.index(bass), False)
        self.window._stop_block((self.index(drums), stem_block(0)))
        heard = self.listen(3.0)
        self.assertSilent(heard, "drums")
        self.assertSounds(heard, "bass")
        self.assertSounds(heard, "other")
        self.assertEqual(self.engine.voice_state((self.index(drums), stem_block(0)))[0], STOPPED)


# The inherited shared-clock tests already run from test_stem_sync.
for _name in [name for name in vars(SharedClockWindowTests) if name.startswith("test_")]:
    setattr(StemIndependenceTests, _name, None)
del SharedClockWindowTests


if __name__ == "__main__":
    unittest.main()
