import os
import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from soniclab.core.stems import (
    STEMS,
    STATUS_CANCELLED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    StemJobQueue,
    StemSplitter,
    existing_stems,
    is_stem_name,
    parse_progress,
)
from soniclab.ui.main_window import MainWindow

RATE = 44100

# Stands in for ``python -m demucs``: the same arguments, progress on stderr
# with carriage returns, and stems under <out>/<model>/<stem>.wav.
FAKE_DEMUCS = textwrap.dedent(
    """
    import sys, time
    import numpy as np, soundfile as sf
    args = sys.argv[3:]
    out = args[args.index("-o") + 1]
    model = args[args.index("-n") + 1]
    track = args[-1]
    delay = float(__import__("os").environ.get("FAKE_DEMUCS_DELAY", "0"))
    sys.stderr.write("Selected model is a bag of 2 models. You will see that many progress bars per track.\\n")
    for _model in range(2):
        for percent in (0, 50, 100):
            sys.stderr.write(f"\\r{percent:3d}%|####| 1/2 [00:00<00:00]")
            sys.stderr.flush()
            time.sleep(delay)
        sys.stderr.write("\\n")
    audio, rate = sf.read(track, dtype="float32", always_2d=True)
    import pathlib
    folder = pathlib.Path(out) / model
    folder.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(("drums", "bass", "other", "vocals")):
        sf.write(folder / f"{stem}.wav", audio * (index + 1) * 0.1, rate)
    """
)


def _fake_python(folder: Path) -> Path:
    script = folder / "fake_demucs.py"
    script.write_text(FAKE_DEMUCS, encoding="utf-8")
    launcher = folder / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


def _song(folder: Path, name="song.wav") -> Path:
    path = folder / name
    t = np.arange(RATE) / RATE
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([tone, tone], axis=1), RATE)
    return path


class ParsingTests(unittest.TestCase):
    def test_progress_spans_every_model_in_the_bag(self):
        state = {}
        self.assertIsNone(parse_progress("Selected model is a bag of 4 models.", state))
        values = [
            parse_progress(line, state)
            for line in (" 0%|", " 50%|##", "100%|####", " 0%|", " 40%|##")
        ]
        self.assertEqual(values, [0.0, 0.125, 0.25, 0.25, 0.35])
        self.assertIsNone(parse_progress("Separating track song.wav", state))

    def test_stem_names_are_not_split_again(self):
        for name in ("drums", "Song vocals", "bass_2", "Instrumental"):
            self.assertTrue(is_stem_name(name), name)
        for name in ("Burma House", "song", "Otherworld", "Piano Man", "Other Side of Town"):
            self.assertFalse(is_stem_name(name), name)


class SplitterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.python = _fake_python(self.root)
        self.song = _song(self.root)

    def tearDown(self):
        os.environ.pop("FAKE_DEMUCS_DELAY", None)
        self.folder.cleanup()

    def test_split_reports_progress_and_reuses_the_folder(self):
        splitter = StemSplitter(self.python)
        seen = []
        result = splitter.split(self.song, self.root / "Stems", on_progress=seen.append)
        self.assertEqual(result.status, STATUS_OK, result.message)
        self.assertEqual(sorted(result.stems), ["bass", "drums", "other", "vocals"])
        for path in result.stems.values():
            self.assertTrue(Path(path).is_file())
            self.assertEqual(Path(path).parent.parent, self.root / "Stems")
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))
        again = splitter.split(self.song, self.root / "Stems")
        self.assertTrue(again.cached)

    def test_plain_demucs_output_beside_the_song_counts_as_split(self):
        folder = self.root / "separated" / "htdemucs_ft" / "song"
        folder.mkdir(parents=True)
        self.assertIsNone(existing_stems(self.song, self.root / "Stems"))
        for stem in STEMS:
            sf.write(folder / f"{stem}.wav", np.zeros((10, 2), np.float32), RATE)
        found = existing_stems(self.song, self.root / "Stems")
        self.assertEqual(found["drums"], str(folder / "drums.wav"))
        result = StemSplitter(self.root / "missing" / "python").split(self.song, self.root / "Stems")
        self.assertTrue(result.cached)

    def test_missing_demucs_is_unavailable(self):
        result = StemSplitter(self.root / "missing" / "python").split(self.song, self.root / "Stems")
        self.assertEqual(result.status, STATUS_UNAVAILABLE)

    def test_cancel_stops_the_running_split(self):
        os.environ["FAKE_DEMUCS_DELAY"] = "0.5"
        queue = StemJobQueue(StemSplitter(self.python))
        job = queue.submit(self.song, self.root / "Stems")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = queue.poll()
            if message and message[0] == "progress":
                break
            time.sleep(0.02)
        queue.cancel()
        queue.drain(5)
        finished = [message for message in iter(queue.poll, None) if message[0] == "finished"]
        self.assertEqual(finished[-1][1], job)
        self.assertEqual(finished[-1][2].status, STATUS_CANCELLED)


class _Jobs:
    def __init__(self):
        self.submitted = []

    def events_for(self, path):
        return ((0.25, 1), (0.75, 1)) if os.path.basename(path or "") == "drums.wav" else ()

    def result_for(self, path):
        return None

    def submit(self, path):
        self.submitted.append(os.path.basename(path))
        return "queued"

    def poll(self):
        return None

    def cancel(self):
        pass


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.window = MainWindow()
        self.window.stem_jobs = StemJobQueue(StemSplitter(_fake_python(self.root)))
        # Signals off: the toggle would otherwise save to the real QSettings.
        self.window.split_on_import_action.blockSignals(True)
        self.window.split_on_import_action.setChecked(True)
        self.window.beat_jobs = self.jobs = _Jobs()
        self.window.current_path = str(self.root / "session.soniclab")
        self.window.show()

    def tearDown(self):
        self.window.stem_jobs.cancel()
        self.window.stem_jobs.drain(5)
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def _wait_for_split(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and self.window._stem_names:
            self.window._poll_stems()
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(self.window._stem_names)
        self.window.finish_loading()

    def test_imported_song_becomes_stems_led_by_drums(self):
        window = self.window
        song = str(_song(self.root, "My Song.wav"))
        window._import_paths([song])
        window.finish_loading()
        self.assertTrue(window.stem_banner.isVisible())
        self.assertIn("Splitting “My Song”", window.stem_label.text())
        self.assertIn("into stems", window.stem_label.toolTip())
        # The full song is never added while it splits.
        self.assertEqual(window.project.tracks, [])
        self.assertEqual(self.jobs.submitted, [])
        self._wait_for_split()
        names = [track["name"] for track in window.project.tracks]
        self.assertEqual(names, ["My Song drums", "My Song bass", "My Song other", "My Song vocals"])
        self.assertTrue(window.project.tracks[0]["drum_reference"])
        self.assertTrue(window.project.tracks[0]["path"].startswith(str(self.root / "Stems")))
        self.assertEqual(self.jobs.submitted, ["drums.wav", "bass.wav", "other.wav", "vocals.wav"])
        self.assertEqual(window._track_events(window.project.tracks[2]), ((0.25, 1), (0.75, 1)))
        self.assertFalse(window.stem_banner.isVisible())
        self.assertIn("into 4 stems", window.statusBar().currentMessage())
        window._undo()
        self.assertEqual(window.project.tracks, [])
        # A second import reuses the stems at once, without Demucs.
        window._import_paths([song])
        window.finish_loading()
        self.assertFalse(window._stem_names)
        self.assertEqual(len(window.project.tracks), 4)
        self.assertIn("Loaded saved stems for “My Song”", window.statusBar().currentMessage())

    def test_failed_split_falls_back_to_the_full_song(self):
        window = self.window
        window.stem_jobs = StemJobQueue(StemSplitter(self.root / "missing" / "python"))
        window.stem_jobs._splitter.python = Path(sys.executable)  # available, but not Demucs
        window._import_paths([str(_song(self.root, "Broken.wav"))])
        window.finish_loading()
        self._wait_for_split()
        self.assertEqual([track["name"] for track in window.project.tracks], ["Broken"])
        self.assertIn("Could not split “Broken”", window.statusBar().currentMessage())

    def test_stem_files_and_the_switch_skip_demucs(self):
        window = self.window
        window._import_paths([str(_song(self.root, "drums.wav"))])
        window.finish_loading()
        self.assertFalse(window._stem_names)
        window.split_on_import_action.setChecked(False)
        window._import_paths([str(_song(self.root, "Another Song.wav"))])
        window.finish_loading()
        self.assertFalse(window._stem_names)
        self.assertEqual(len(window.project.tracks), 2)


if __name__ == "__main__":
    unittest.main()
