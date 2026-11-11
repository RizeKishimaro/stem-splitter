import os
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

from soniclab.core.downbeat_transfer import (
    DownbeatFileError,
    FILE_SUFFIX,
    default_folder,
    export_document,
    import_events,
    read_document,
    track_events,
    transfer_events,
    write_document,
)
from soniclab.core.project import Project
from soniclab.ui.main_window import MainWindow


class _Jobs:
    def __init__(self, events):
        self.events = tuple(events)

    def events_for(self, path):
        return self.events if path == "/tmp/soniclab-source.wav" else ()

    def result_for(self, path):
        return None

    def submit(self, path):
        return "missing"

    def poll(self):
        return None

    def cancel(self):
        pass


class TransferTests(unittest.TestCase):
    def test_transfer_keeps_arrangement_time(self):
        # Source clip at 4.0 s plays its file from 1.0 s, so source 3.0 is 6.0 s.
        moved = transfer_events([(1.0, 1), (3.0, 1), (3.5, 2)], 4.0, 1.0, 5.0, 0.5)
        self.assertEqual(moved, [[1.5, 1], [2.0, 2]])

    def test_beats_before_the_target_file_are_dropped(self):
        self.assertEqual(transfer_events([(0.5, 1), (2.0, 1)], 0.0, 0.0, 1.0, 0.0), [[1.0, 1]])

    def test_applied_list_overrides_detected(self):
        self.assertEqual(track_events({}, [(1.0, 1)]), ((1.0, 1),))
        self.assertEqual(track_events({"downbeats": [[2.0, 1]]}, [(1.0, 1)]), ((2.0, 1),))

    def test_default_folder_sits_beside_a_saved_project(self):
        self.assertEqual(default_folder("/x/song/song.soniclab"), Path("/x/song/Downbeats"))
        self.assertEqual(default_folder().name, "downbeats")


class FileTests(unittest.TestCase):
    def test_round_trip_lines_up_with_target_clip_edge(self):
        document = export_document([(0.5, 1), (1.5, 2), (2.5, 1), (9.0, 1)], 0.5, 4.0, name="Kick")
        self.assertEqual(
            [(item["local_seconds"], item["beat_position"]) for item in document["events"]],
            [(0.0, 1), (1.0, 2), (2.0, 1)],
        )
        with tempfile.TemporaryDirectory() as folder:
            written = write_document(Path(folder) / "sub" / "kick.json", document)
            self.assertTrue(written.name.endswith(FILE_SUFFIX))
            loaded = read_document(written)
        self.assertEqual(import_events(loaded, 3.0), [[3.0, 1], [4.0, 2], [5.0, 1]])

    def test_rejects_other_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"
            path.write_text('{"tracks": []}', encoding="utf-8")
            with self.assertRaises(DownbeatFileError):
                read_document(path)
            path.write_text('{"format": "soniclab-downbeats", "version": 1, "events": [{"local_seconds": 1, "beat_position": 0}]}', encoding="utf-8")
            with self.assertRaises(DownbeatFileError):
                read_document(path)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs([(1.0, 1), (2.0, 2), (3.0, 1), (5.0, 1)])
        audio = np.zeros((128, 2), dtype=np.float32)
        for name, path, start in (
            ("Drums", "/tmp/soniclab-source.wav", 0.0),
            ("Bass", "/tmp/soniclab-bass.wav", 2.0),
        ):
            self.window.project.tracks.append(
                {"name": name, "path": path, "start": start, "offset": 0.0, "duration": 6.0, "_audio": audio}
            )
            self.window.engine.add_clip(audio, start)
        self.window._refresh_timeline()
        self.window._select_track(0)

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_apply_undo_and_save(self):
        window = self.window
        window._apply_downbeats_to_tracks([1])
        bass = window.project.tracks[1]
        self.assertEqual(bass["downbeats"], [[0.0, 2], [1.0, 1], [3.0, 1]])
        self.assertEqual(window._track_events(bass), ((0.0, 2), (1.0, 1), (3.0, 1)))
        window._undo()
        self.assertNotIn("downbeats", window.project.tracks[1])
        window._redo()
        self.assertIn("downbeats", window.project.tracks[1])
        with tempfile.TemporaryDirectory() as folder:
            window.current_path = os.path.join(folder, "song.soniclab")
            window._save_project()
            saved = Project.load(window.current_path)
        self.assertEqual(saved.tracks[1]["downbeats"], [[0.0, 2], [1.0, 1], [3.0, 1]])

    def test_menu_actions_open_their_dialogs(self):
        window = self.window
        asked = []
        window._downbeat_targets = lambda source: asked.append(source) or [1]
        window.apply_downbeats_action.trigger()
        self.assertEqual(asked, [0])
        self.assertEqual(window.project.tracks[1]["downbeats"], [[0.0, 2], [1.0, 1], [3.0, 1]])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "drums" + FILE_SUFFIX)
            with mock.patch(
                "soniclab.ui.main_window.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                window.export_downbeats_action.trigger()
            self.assertTrue(os.path.exists(path))
            window._select_track(1)
            window._clear_applied_downbeats()
            with mock.patch(
                "soniclab.ui.main_window.QFileDialog.getOpenFileName", return_value=(path, "")
            ):
                window.import_downbeats_action.trigger()
        self.assertEqual(window.project.tracks[1]["downbeats"], [[1.0, 1], [2.0, 2], [3.0, 1], [5.0, 1]])

    def test_export_then_import_into_another_track(self):
        window = self.window
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "drums" + FILE_SUFFIX)
            window._export_downbeats(path)
            window._select_track(1)
            window._import_downbeats(path)
        self.assertEqual(window.project.tracks[1]["downbeats"], [[1.0, 1], [2.0, 2], [3.0, 1], [5.0, 1]])
        window._clear_applied_downbeats()
        self.assertNotIn("downbeats", window.project.tracks[1])


if __name__ == "__main__":
    unittest.main()
