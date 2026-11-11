import os
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from soniclab.ui.main_window import MainWindow


class ArrowKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        audio = np.zeros((128, 2), dtype=np.float32)
        self.window.project.tracks.append(
            {"name": "Song", "path": None, "start": 0.0, "offset": 0.0, "duration": 12.0, "_audio": audio}
        )
        self.window.engine.add_clip(audio, 0.0)
        self.window._refresh_timeline()
        self.window.show()
        self.window.timeline.setFocus()
        self.app.processEvents()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _press(self, key, widget=None, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(widget or self.window.timeline, key, modifier)
        self.app.processEvents()

    def test_left_and_right_seek_five_seconds(self):
        window = self.window
        self._press(Qt.Key.Key_Right)
        self.assertAlmostEqual(window.engine.position, 5.0)
        self._press(Qt.Key.Key_Right)
        self._press(Qt.Key.Key_Right)
        self.assertAlmostEqual(window.engine.position, 12.0)  # stops at the end
        self._press(Qt.Key.Key_Left)
        self.assertAlmostEqual(window.engine.position, 7.0)
        self._press(Qt.Key.Key_Left)
        self._press(Qt.Key.Key_Left)
        self.assertAlmostEqual(window.engine.position, 0.0)

    def test_up_and_down_step_master_volume(self):
        window = self.window
        self._press(Qt.Key.Key_Down)
        self._press(Qt.Key.Key_Down)
        self.assertAlmostEqual(window.engine._output_gain, 0.8)
        self.assertIn("80%", window.statusBar().currentMessage())
        self._press(Qt.Key.Key_Up)
        self.assertAlmostEqual(window.engine._output_gain, 0.9)
        for _ in range(12):
            self._press(Qt.Key.Key_Down)
        self.assertEqual(window.engine._output_gain, 0.0)
        window._undo()  # one key press per undo step
        self.assertAlmostEqual(window.engine._output_gain, 0.1)

    def test_text_fields_and_modifiers_keep_their_arrows(self):
        window = self.window
        window.project_name.setFocus()
        self._press(Qt.Key.Key_Right, window.project_name)
        self._press(Qt.Key.Key_Up, window.project_name)
        self._press(Qt.Key.Key_Right, modifier=Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(window.engine.position, 0.0)
        self.assertEqual(window.engine._output_gain, 1.0)


if __name__ == "__main__":
    unittest.main()
