import os
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from soniclab.core.drums import name_says_drums, percussive_share, pick_drum_track
from soniclab.ui.main_window import MainWindow

RATE = 44100


def _drums(seconds=12.0):
    rng = np.random.default_rng(1)
    audio = np.zeros(int(seconds * RATE), dtype=np.float32)
    burst = int(0.08 * RATE)
    envelope = np.exp(-np.linspace(0, 8, burst)).astype(np.float32)
    for start in range(0, audio.size - burst, int(0.25 * RATE)):
        audio[start : start + burst] += 0.6 * rng.standard_normal(burst).astype(np.float32) * envelope
    return np.stack([audio, audio], axis=1)


def _pad(seconds=12.0):
    t = np.arange(int(seconds * RATE)) / RATE
    audio = 0.2 * (np.sin(2 * np.pi * 110 * t) + np.sin(2 * np.pi * 220 * t))
    return np.stack([audio, audio], axis=1).astype(np.float32)


class _Jobs:
    def __init__(self, events_by_name):
        self.events_by_name = events_by_name
        self.submitted = []

    def events_for(self, path):
        return self.events_by_name.get(os.path.basename(path or ""), ())

    def result_for(self, path):
        return None

    def submit(self, path):
        self.submitted.append(os.path.basename(path))
        return "queued"

    def poll(self):
        return None

    def cancel(self):
        pass


class DetectionTests(unittest.TestCase):
    def test_names(self):
        for name in ("drums", "Drum Loop", "perc_2", "Kick", "OH overheads", "liveKit", "HiHat"):
            self.assertTrue(name_says_drums(name), name)
        for name in ("bass", "vocals", "other", "Kitchen", "Tomato", "Song"):
            self.assertFalse(name_says_drums(name), name)

    def test_noise_bursts_beat_a_held_chord(self):
        self.assertGreater(percussive_share(_drums(), RATE), 0.45)
        self.assertLess(percussive_share(_pad(), RATE), 0.3)
        self.assertEqual(percussive_share(np.zeros((RATE * 4, 2), np.float32), RATE), 0.0)

    def test_pick_prefers_marked_then_name_then_sound(self):
        pad, drums = _pad(4.0), _drums(4.0)
        tracks = [{"name": "a", "_audio": pad}, {"name": "b", "_audio": drums}]
        self.assertEqual(pick_drum_track(tracks, RATE), 1)
        self.assertIsNotNone(tracks[0].get("_drum_score"))
        tracks = [{"name": "a", "_audio": drums}, {"name": "Drums", "_audio": pad}]
        self.assertEqual(pick_drum_track(tracks, RATE), 1)
        tracks[0]["drum_reference"] = True
        self.assertEqual(pick_drum_track(tracks, RATE), 0)
        self.assertIsNone(pick_drum_track([{"name": "a", "_audio": pad}], RATE))


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.paths = {}
        for name, audio in (("bass.wav", _pad(4.0)), ("drums.wav", _drums(4.0))):
            path = os.path.join(self.folder.name, name)
            sf.write(path, audio, RATE)
            self.paths[name] = path
        self.window = MainWindow()
        self.jobs = _Jobs({"drums.wav": ((0.5, 1), (1.0, 2), (2.5, 1)), "bass.wav": ((0.1, 1),)})
        self.window.beat_jobs = self.jobs

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def test_import_marks_drums_and_others_follow(self):
        window = self.window
        window._import_paths([self.paths["bass.wav"], self.paths["drums.wav"]])
        bass, drums = window.project.tracks
        self.assertTrue(drums.get("drum_reference"))
        self.assertEqual(self.jobs.submitted, ["drums.wav", "bass.wav"])
        self.assertIn("Drum track: drums", window.statusBar().currentMessage())
        self.assertEqual(window._track_events(bass), ((0.5, 1), (1.0, 2), (2.5, 1)))
        # Moving the bass keeps the markers at the drum's arrangement times.
        bass["start"] = 1.0
        self.assertEqual(window._track_events(bass), ((0.0, 2), (1.5, 1)))
        window._undo()
        self.assertEqual(window.project.tracks, [])

    def test_toggle_drum_track(self):
        window = self.window
        window._import_paths([self.paths["bass.wav"], self.paths["drums.wav"]])
        window._select_track(0)
        window._toggle_drum_track()
        self.assertTrue(window.project.tracks[0].get("drum_reference"))
        self.assertFalse(window.project.tracks[1].get("drum_reference"))
        self.assertEqual(window._track_events(window.project.tracks[1]), ((0.1, 1),))
        window._toggle_drum_track()
        self.assertIsNone(window._drum_index())
        self.assertEqual(window._track_events(window.project.tracks[0]), ((0.1, 1),))

    def test_dropping_files_on_the_timeline_imports_them(self):
        window = self.window
        window.show()
        self.app.processEvents()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(self.paths["drums.wav"]), QUrl.fromLocalFile("/tmp/notes.txt")])
        event = QDropEvent(
            QPointF(20, 20),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.assertTrue(window.eventFilter(window.timeline.viewport(), event))
        self.app.processEvents()
        self.assertEqual([track["name"] for track in window.project.tracks], ["drums"])


if __name__ == "__main__":
    unittest.main()
