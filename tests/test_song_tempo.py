"""The BPM field with a song: it shows the tempo of the downbeats, and a new
tempo time-stretches every track (pitch kept) with positions, bars and
beats scaled to match."""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SONICLAB_DEMUCS_PYTHON", "")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.ui.main_window import MainWindow


class _Jobs:
    """Beat analysis stand-in: the drum file has a downbeat every 2 s."""

    def __init__(self, drum_path):
        self.drum_path = str(drum_path)

    def events_for(self, path):
        if str(path) != self.drum_path:
            return ()
        return tuple((t, 1 + i % 4) for i, t in enumerate(np.arange(0.0, 16.0, 0.5)))

    def result_for(self, path):
        return None

    def submit(self, path):
        return "queued"

    def poll(self):
        return None

    def cancel(self):
        pass


def song(folder: Path):
    """Drums: a click on every beat at 120 BPM, loud on bar 3 (4-6 s).
    Bass: a 110 Hz tone."""
    t = np.arange(OUTPUT_RATE * 16) / OUTPUT_RATE
    drums = np.zeros_like(t)
    for beat in np.arange(0.0, 16.0, 0.5):
        at = int(beat * OUTPUT_RATE)
        drums[at : at + 2000] += np.exp(-np.arange(2000) / 300.0) * (0.8 if 4.0 <= beat < 6.0 else 0.3)
    bass = 0.2 * np.sin(2 * np.pi * 110 * t)
    paths = []
    for name, signal in (("drums", drums), ("bass", bass)):
        path = folder / f"{name}.wav"
        sf.write(path, np.stack([signal, signal], 1).astype(np.float32), OUTPUT_RATE)
        paths.append(path)
    return paths


class SongTempoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        drums, bass = song(self.root)
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs(drums)
        self.window._import_paths([str(drums), str(bass)], split=False)
        self.window.project.tracks[0]["drum_reference"] = True
        self.window._refresh_timeline()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def level(self, seconds, width=0.05):
        audio = self.window.engine.render(int(seconds * OUTPUT_RATE), int(width * OUTPUT_RATE))
        return float(np.abs(audio).max())

    def test_field_shows_the_tempo_of_the_downbeats(self):
        self.assertAlmostEqual(self.window._song_tempo(), 120.0, places=3)
        self.assertAlmostEqual(self.window.bpm.value(), 120.0)
        self.assertIn("Type a new tempo", self.window.bpm.toolTip())

    def test_new_tempo_stretches_the_song_and_keeps_it_on_the_grid(self):
        window = self.window
        drums, bass = window.project.tracks
        bars_before = window._bar_lines()
        self.assertTrue(window.change_tempo(100.0))  # 1.2 x longer
        self.assertAlmostEqual(window.bpm.value(), 100.0)
        self.assertAlmostEqual(window._song_tempo(), 100.0, places=2)
        for track in (drums, bass):
            self.assertAlmostEqual(track["stretch"], 1.2)
            self.assertAlmostEqual(track["duration"], 16.0 * 1.2, places=3)
            self.assertEqual(len(track["_audio"]), round(16 * OUTPUT_RATE * 1.2))
            self.assertEqual(len(track["_source_audio"]), 16 * OUTPUT_RATE)
        # Bars follow the drums' downbeats, which scaled with the audio.
        self.assertAlmostEqual(window._bar_lines()[2], bars_before[2] * 1.2, places=3)
        # The loud bar (4-6 s) now plays from 4.8 to 7.2 s, between quieter bars.
        def energy(audio, start, end):
            part = audio[int(start * OUTPUT_RATE) : int(end * OUTPUT_RATE), 0]
            return float(np.mean(part**2))

        # Windows stay 50 ms inside the bar edges: a hit may move by up to 12 ms.
        stretched = drums["_audio"]
        loud = energy(stretched, 4.85, 7.15)
        self.assertGreater(loud, 4 * energy(stretched, 2.45, 4.75))
        self.assertGreater(loud, 4 * energy(stretched, 7.25, 9.55))
        # Pitch: the bass is still 110 Hz.
        bass_only = bass["_audio"][int(8 * OUTPUT_RATE) : int(8 * OUTPUT_RATE) + 16384, 0]
        spectrum = np.abs(np.fft.rfft(bass_only * np.hanning(len(bass_only))))
        self.assertAlmostEqual(np.argmax(spectrum) * OUTPUT_RATE / len(bass_only), 110.0, delta=3.0)

        # A second change stretches from the original audio, not the result.
        window.change_tempo(150.0)
        self.assertAlmostEqual(drums["stretch"], 0.8)
        self.assertEqual(len(drums["_audio"]), round(16 * OUTPUT_RATE * 0.8))
        # One undo per change.
        window._undo()
        self.assertAlmostEqual(window.project.tracks[0]["stretch"], 1.2)
        self.assertAlmostEqual(window.bpm.value(), 100.0)
        window._undo()
        self.assertNotIn("stretch", window.project.tracks[0])
        self.assertEqual(len(window.project.tracks[0]["_audio"]), 16 * OUTPUT_RATE)
        self.assertAlmostEqual(window.bpm.value(), 120.0)

    def test_typing_a_tempo_applies_it_after_a_pause(self):
        window = self.window
        window.bpm.setValue(130.0)
        self.assertNotIn("stretch", window.project.tracks[0])  # not on every keystroke
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "stretch" not in window.project.tracks[0]:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertAlmostEqual(window.project.tracks[0]["stretch"], 120.0 / 130.0)
        self.assertAlmostEqual(window.bpm.value(), 130.0)

    def test_saved_tempo_comes_back_on_open(self):
        window = self.window
        window.change_tempo(96.0)
        path = self.root / "session.soniclab"
        window.current_path = str(path)
        window._save_project()
        other = MainWindow()
        other.beat_jobs = window.beat_jobs
        try:
            other._load_project(str(path))
            track = other.project.tracks[0]
            self.assertAlmostEqual(track["stretch"], 1.25)
            self.assertEqual(len(track["_audio"]), round(16 * OUTPUT_RATE * 1.25))
            self.assertAlmostEqual(other._song_tempo(), 96.0, places=2)
        finally:
            other.close()
            other.deleteLater()

    def test_even_bars_at_the_typed_tempo(self):
        window = self.window
        window.bpm.blockSignals(True)
        window.bpm.setValue(90.0)
        window.bpm.blockSignals(False)
        window._even_bars_at_tempo()
        lines = window.project.bars
        self.assertAlmostEqual(lines[1] - lines[0], 60.0 * 4 / 90.0, places=4)
        self.assertNotIn("stretch", window.project.tracks[0])  # audio unchanged


if __name__ == "__main__":
    unittest.main()
