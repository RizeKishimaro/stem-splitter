import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from soniclab.core.alignment import (
    KIND_EARLY,
    KIND_LATE,
    KIND_ON_TIME,
    KIND_UNMATCHED,
    arrangement_time,
    classify_onsets,
    nudged_alignment,
    reference_beats,
)
from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.core.onsets import STATUS_OK, OnsetResult, analyze_path, detect_onsets
from soniclab.core.onset_jobs import OnsetJobQueue
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import ClipItem


REFERENCE = (
    (0.0, 1),
    (0.5, 2),
    (1.0, 3),
    (1.5, 4),
    (2.0, 1),
)


def _click_audio(rate=OUTPUT_RATE):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(rate * 4) * 0.001).astype(np.float32)
    hits = (1.0, 1.5, 2.0, 2.5, 3.20, 3.23)
    for hit in hits:
        start = int(hit * rate)
        audio[start] = 1.0
        audio[start + 1 : start + 40] += np.linspace(0.8, 0.0, 39, dtype=np.float32)
    return np.column_stack((audio, audio)), hits


class OnsetDetectionTests(unittest.TestCase):
    def test_complex_onsets_follow_the_clicks_and_keep_the_gap(self):
        audio, hits = _click_audio()
        detected = detect_onsets(audio, OUTPUT_RATE)
        self.assertTrue(all(time > 0.0 for time in detected))
        for earlier, later in zip(detected, detected[1:]):
            self.assertGreaterEqual(later - earlier, 0.05 - 1e-3)
        for hit in hits[:4]:
            self.assertLess(min(abs(time - hit) for time in detected), 0.04)
        self.assertFalse(any(abs(time - 3.23) < 0.02 for time in detected))
        self.assertEqual(detect_onsets(np.zeros(OUTPUT_RATE, dtype=np.float32), OUTPUT_RATE), ())

    def test_source_file_is_cached_by_hash(self):
        audio, _hits = _click_audio()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clicks.wav"
            sf.write(source, audio, OUTPUT_RATE)
            cache = root / "cache"
            first = analyze_path(source, cache)
            second = analyze_path(source, cache)
        self.assertEqual(first.status, STATUS_OK)
        self.assertGreaterEqual(len(first.onsets), 4)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(second.onsets, first.onsets)


class MappingTests(unittest.TestCase):
    def test_source_offset_and_alignment_share_one_origin(self):
        self.assertAlmostEqual(arrangement_time(2.0, 10.0, 0.5, -0.02), 11.48)
        events = list(REFERENCE)
        beats = reference_beats(events, 10.0, 0.5, -0.02)
        self.assertEqual([position for _time, position in beats], [1, 2, 3, 4, 1])
        self.assertAlmostEqual(beats[1][0], 10.0 + (0.5 - 0.5) - 0.02)
        self.assertEqual(events, list(REFERENCE))

    def test_confident_matches_use_every_beat_position(self):
        onsets = [0.47, 0.53, 0.51, 0.90]
        marks = classify_onsets(onsets, REFERENCE, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0)
        kinds = {round(mark.source_seconds, 2): mark.kind for mark in marks}
        self.assertEqual(kinds[0.47], KIND_EARLY)
        self.assertEqual(kinds[0.53], KIND_LATE)
        self.assertEqual(kinds[0.51], KIND_ON_TIME)
        self.assertEqual(kinds[0.90], KIND_UNMATCHED)
        late = next(mark for mark in marks if mark.source_seconds == 0.53)
        self.assertAlmostEqual(late.error_seconds, 0.03)
        self.assertAlmostEqual(late.local_seconds, 0.53)
        unmatched = next(mark for mark in marks if mark.source_seconds == 0.90)
        self.assertIsNone(unmatched.error_seconds)

    def test_alignment_changes_the_error_without_moving_the_onset(self):
        before = classify_onsets((0.53,), REFERENCE, 0.25, 0.25, 0.0, 2.0, 0.0, 0.0, 0.0)
        after = classify_onsets((0.53,), REFERENCE, 0.25, 0.25, -0.03, 2.0, 0.0, 0.0, 0.0)
        self.assertEqual(before[0].kind, KIND_LATE)
        self.assertEqual(before[0].local_seconds, after[0].local_seconds)
        self.assertAlmostEqual(before[0].local_seconds, 0.28)
        self.assertEqual(after[0].kind, KIND_ON_TIME)
        self.assertAlmostEqual(after[0].error_seconds, 0.0)

    def test_a_tight_beat_gap_rejects_a_weak_association(self):
        beats = ((0.0, 1), (0.10, 2))
        marks = classify_onsets((0.16,), beats, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        self.assertEqual(marks[0].kind, KIND_UNMATCHED)

    def test_nudge_uses_only_confident_errors(self):
        marks = classify_onsets((0.53, 0.55, 0.90), REFERENCE, 0.0, 0.0, 0.01, 3.0, 0.0, 0.0, 0.0)
        errors = [mark.error_seconds for mark in marks if mark.error_seconds is not None]
        self.assertEqual(len(errors), 2)
        updated = nudged_alignment(0.01, errors)
        self.assertAlmostEqual(updated, 0.01 - 0.05)
        self.assertIsNone(nudged_alignment(0.0, ()))


class _Analyzer:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.block = threading.Event()

    def __call__(self, path, cancel_event=None):
        self.calls.append(str(path))
        self.entered.set()
        self.block.wait(3)
        return OnsetResult(STATUS_OK, (0.4, 0.9))


class QueueTests(unittest.TestCase):
    def test_identical_sources_share_one_onset_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.wav"
            second = root / "b.wav"
            first.write_bytes(b"same-onset-bytes")
            second.write_bytes(b"same-onset-bytes")
            analyzer = _Analyzer()
            queue = OnsetJobQueue(analyzer)
            self.assertEqual(queue.submit(first), "queued")
            self.assertTrue(analyzer.entered.wait(2))
            self.assertEqual(queue.submit(second), "queued")
            analyzer.block.set()
            queue.drain()
            self.assertEqual(len(analyzer.calls), 1)
            self.assertEqual(queue.onsets_for(first), (0.4, 0.9))
            self.assertEqual(queue.onsets_for(second), (0.4, 0.9))
            self.assertEqual(queue.submit(second), "ready")
            self.assertEqual(len(analyzer.calls), 1)


class PaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_onset_colors_follow_the_kind_and_leave_aubio_marks(self):
        clip = ClipItem(0, "Bass", np.zeros((128, 2), dtype=np.float32), 2.0, "#6E93B0", None, 22.5, 90.0)
        clip.set_beats((0.2, 0.8), 0.0, 4, True)
        aubio = clip.beat_markers()
        clip.set_onsets(((0.4, KIND_LATE), (1.2, KIND_UNMATCHED)), True)
        self.assertEqual(clip.beat_markers(), aubio)
        image = QImage(int(clip.rect().width()), int(clip.rect().height()), QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        clip._paint_onsets(painter, 0, clip.rect().width())
        painter.end()
        late = image.pixelColor(int(round(0.4 * 90)), 28)
        unmatched = image.pixelColor(int(round(1.2 * 90)), 28)
        self.assertGreater(late.red(), late.blue() + 20)
        self.assertGreater(unmatched.alpha(), 40)
        self.assertLess(abs(unmatched.red() - unmatched.blue()), 30)


class _Events:
    def __init__(self, events):
        self.events = tuple(events)

    def events_for(self, _path):
        return self.events

    def poll(self):
        return None

    def submit(self, _path):
        return "ready"

    def result_for(self, _path):
        return None

    def cancel(self):
        return None


class _Onsets:
    def __init__(self, onsets):
        self.onsets = tuple(onsets)

    def onsets_for(self, _path):
        return self.onsets

    def poll(self):
        return None

    def submit(self, _path):
        return "ready"

    def cancel(self):
        return None


class NudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_nudge_changes_alignment_only_and_can_be_undone(self):
        self.window.beat_jobs = _Events(REFERENCE)
        self.window.onset_jobs = _Onsets((0.53, 0.90))
        audio = np.zeros((128, 2), dtype=np.float32)
        self.window.project.tracks.extend(
            [
                {"name": "Drums", "path": "/tmp/drums.wav", "start": 0.0, "offset": 0.0, "duration": 3.0, "_audio": audio},
                {"name": "Bass", "path": "/tmp/bass.wav", "start": 1.0, "offset": 0.0, "duration": 3.0, "_audio": audio},
            ]
        )
        self.window._refresh_timeline()
        self.window._select_track(0)
        self.window._set_beat_reference()
        self.window._select_track(1)
        bpm = self.window.project.bpm
        signature = list(self.window.project.time_signature)
        snap = self.window.timeline.snap_to_grid
        loop = self.window.engine.loop
        self.window._nudge_to_reference()
        bass = self.window.project.tracks[1]
        self.assertAlmostEqual(bass["_alignment"], -0.03)
        self.assertEqual(bass["start"], 1.0)
        self.assertEqual(bass["offset"], 0.0)
        self.assertEqual(self.window.onset_jobs.onsets, (0.53, 0.90))
        self.assertEqual(self.window.project.bpm, bpm)
        self.assertEqual(self.window.project.time_signature, signature)
        self.assertEqual(self.window.timeline.snap_to_grid, snap)
        self.assertEqual(self.window.engine.loop, loop)
        nudged = {round(mark.source_seconds, 2): mark.kind for mark in self.window._onset_marks(1)}
        self.assertEqual(nudged[0.53], KIND_ON_TIME)
        self.assertEqual(nudged[0.90], KIND_UNMATCHED)
        self.window._undo()
        self.assertEqual(self.window.project.tracks[1].get("_alignment", 0.0), 0.0)
        self.assertEqual(self.window.project.tracks[1]["start"], 1.0)
