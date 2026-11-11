"""The public AudioEngine surface the UI uses, and that a native engine must match."""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import MainWindow

UI_DIR = Path(__file__).resolve().parent.parent / "soniclab" / "ui"


def _tone(frames=256, value=0.5):
    return np.full((frames, 2), value, dtype=np.float32)


class EnginePublicInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()

    def tearDown(self):
        self.engine.close()

    def test_defaults(self):
        self.assertEqual(self.engine.output_gain, 1.0)
        self.assertIs(self.engine.master_muted, False)
        self.assertEqual(self.engine.clip_count, 0)
        self.assertIs(self.engine.playing, False)

    def test_add_clip_returns_its_index(self):
        self.assertEqual(self.engine.add_clip(_tone(), 0.0), 0)
        self.assertEqual(self.engine.add_clip(_tone(), 1.0), 1)
        self.assertEqual(self.engine.clip_count, 2)

    def test_returned_index_addresses_the_clip(self):
        self.engine.add_clip(_tone(value=0.25), 0.0)
        index = self.engine.add_clip(_tone(value=0.5), 0.0)
        self.engine.set_clip_muted(index, True)
        mixed = self.engine.render(0, 16)
        self.assertTrue(np.allclose(mixed, 0.25))

    def test_output_gain_reflects_setter_and_clamp(self):
        self.engine.set_output_gain(0.8)
        self.assertAlmostEqual(self.engine.output_gain, 0.8)
        self.assertIsInstance(self.engine.output_gain, float)
        self.engine.set_output_gain(5.0)
        self.assertEqual(self.engine.output_gain, 1.5)
        self.engine.set_output_gain(-1.0)
        self.assertEqual(self.engine.output_gain, 0.0)

    def test_master_muted_reflects_setter(self):
        self.engine.set_master_muted(1)
        self.assertIs(self.engine.master_muted, True)
        self.engine.add_clip(_tone(), 0.0)
        self.assertTrue(np.all(self.engine.render(0, 16) == 0.0))
        self.engine.set_master_muted(False)
        self.assertIs(self.engine.master_muted, False)

    def test_clear_clips_keeps_transport_loop_and_master(self):
        self.engine.add_clip(_tone(), 0.0)
        self.engine.add_clip(_tone(), 0.0)
        self.engine.seek(100 / OUTPUT_RATE)
        self.engine.set_loop(0.0, 200 / OUTPUT_RATE)
        self.engine.set_output_gain(0.5)
        self.engine.clear_clips()
        self.assertEqual(self.engine.clip_count, 0)
        self.assertAlmostEqual(self.engine.position, 100 / OUTPUT_RATE)
        self.assertEqual(self.engine.loop, (0.0, 200 / OUTPUT_RATE))
        self.assertEqual(self.engine.output_gain, 0.5)
        self.assertTrue(np.all(self.engine.render(0, 16) == 0.0))
        self.assertEqual(len(self.engine.read_track(0, 16)), 0)
        self.assertEqual(self.engine.add_clip(_tone(), 0.0), 0)

    def test_properties_do_not_expose_mutable_state(self):
        self.engine.add_clip(_tone(), 0.0)
        count = self.engine.clip_count
        count += 1
        self.assertEqual(self.engine.clip_count, 1)


class WindowUsesPublicInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_ui_never_reads_private_engine_fields(self):
        pattern = re.compile(r"\bengine\._[A-Za-z]")
        offenders = []
        for path in sorted(UI_DIR.glob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_snapshot_reads_master_state_through_properties(self):
        self.window.engine.set_output_gain(0.7)
        self.window.engine.set_master_muted(True)
        snap = self.window._snapshot()
        self.assertAlmostEqual(snap["master_gain"], 0.7)
        self.assertIs(snap["master_muted"], True)

    def test_master_volume_step_starts_from_public_gain(self):
        self.window.engine.set_output_gain(0.5)
        self.window._step_master_volume(0.1)
        self.assertAlmostEqual(self.window.engine.output_gain, 0.6)

    def test_added_track_and_clip_indices_stay_aligned(self):
        with mock.patch("soniclab.ui.main_window.QInputDialog.getText", return_value=("Vox", True)):
            self.window._add_empty_track()
            self.window._add_empty_track()
        self.assertEqual(len(self.window.project.tracks), 2)
        self.assertEqual(self.window.engine.clip_count, len(self.window.project.tracks))
