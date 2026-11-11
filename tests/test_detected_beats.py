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
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from soniclab.core.beat_analysis import (
    STATUS_CANCELLED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    BeatAnalysisResult,
    BeatEvent,
)
from soniclab.core.beat_jobs import BeatJobQueue
from soniclab.ui.timeline import ClipItem, clip_local_beats


class _Client:
    def __init__(self, result=None, block=None):
        self.result = result or BeatAnalysisResult(
            status=STATUS_OK,
            events=(BeatEvent(0.5, 1), BeatEvent(1.0, 2), BeatEvent(1.5, 1)),
        )
        self.block = block
        self.calls = []
        self.entered = threading.Event()

    def analyze_path(self, path, fps=100, beats_per_bar=(3, 4), on_progress=None, cancel_event=None):
        self.calls.append((str(path), fps, tuple(beats_per_bar)))
        self.entered.set()
        if self.block is not None:
            self.block.wait(3)
        if on_progress is not None:
            on_progress("detecting")
        return self.result


class CoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_local_seconds_use_offset_and_exclude_the_duration_edge(self):
        events = [(0.4, 1), (0.5, 1), (1.0, 2), (1.5, 1), (2.0, 4)]
        marks = clip_local_beats(events, 0.5, 1.5)
        self.assertEqual(marks, [(0.0, True), (0.5, False), (1.0, True)])

    def test_downbeats_are_only_beat_position_one(self):
        marks = clip_local_beats([(0.0, 1), (0.5, 2), (1.0, 3), (1.5, 4)], 0.0, 2.0)
        self.assertEqual([downbeat for _local, downbeat in marks], [True, False, False, False])

    def test_clip_follows_move_trim_and_zoom_from_timestamps(self):
        audio = np.zeros((128, 2), dtype=np.float32)
        clip = ClipItem(0, "Bass", audio, 1.5, "#6E93B0", None, 22.5, 90.0)
        events = [(0.5, 1), (1.0, 2), (1.5, 1), (2.0, 4)]
        clip.set_beats((0.2, 0.55, 1.4), 0.0, 4, True)
        aubio = clip.beat_markers()
        clip.set_detected_beats(events, 0.5, True)
        self.assertEqual(clip.detected_markers(), [(0.0, True), (0.5, False), (1.0, True)])
        self.assertEqual(clip.beat_markers(), aubio)
        clip.setPos(480, clip.pos().y() + 40)
        self.assertEqual(clip.detected_markers(), [(0.0, True), (0.5, False), (1.0, True)])
        clip.pixels_per_second = 180.0
        self.assertEqual([local for local, _downbeat in clip.detected_markers()], [0.0, 0.5, 1.0])
        self.assertEqual(int(round(0.5 * clip.pixels_per_second)), 90)
        clip.set_detected_beats(events, 1.25, True)
        self.assertEqual(clip.detected_markers(), [(0.25, True), (0.75, False)])
        clip.set_detected_beats(events, 0.5, False)
        self.assertEqual(clip.detected_markers(), [])
        self.assertEqual(clip.beat_markers(), aubio)

    def test_downbeat_paints_a_triangle(self):
        audio = np.zeros((128, 2), dtype=np.float32)
        clip = ClipItem(0, "Bass", audio, 2.0, "#6E93B0", None, 22.5, 90.0)
        clip.set_detected_beats([(0.5, 1), (1.0, 2)], 0.0, True)
        image = QImage(int(clip.rect().width()), int(clip.rect().height()), QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        clip._paint_detected(painter, 0, clip.rect().width())
        painter.end()
        downbeat_x = int(round(0.5 * clip.pixels_per_second))
        beat_x = int(round(1.0 * clip.pixels_per_second))

        def gold_span(x, y):
            left = right = x
            while left > 0 and image.pixelColor(left - 1, y).alpha() > 40:
                left -= 1
            while right < image.width() - 1 and image.pixelColor(right + 1, y).alpha() > 40:
                right += 1
            return right - left

        self.assertGreater(image.pixelColor(downbeat_x, 3).alpha(), 40)
        self.assertGreater(gold_span(downbeat_x, 3), 4)
        self.assertEqual(image.pixelColor(beat_x, 4).alpha(), 0)
        self.assertEqual(image.pixelColor(beat_x, 3).alpha(), 0)


class JobTests(unittest.TestCase):
    def test_identical_sources_share_one_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.wav"
            second = root / "b.wav"
            first.write_bytes(b"same-source-bytes")
            second.write_bytes(b"same-source-bytes")
            gate = threading.Event()
            client = _Client(block=gate)
            queue = BeatJobQueue(client)
            self.assertEqual(queue.submit(first), "queued")
            self.assertTrue(client.entered.wait(2))
            self.assertEqual(queue.submit(first), "duplicate")
            self.assertEqual(queue.submit(second), "queued")
            gate.set()
            queue.drain()
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0][1:], (100, (3, 4)))
            self.assertEqual(queue.events_for(first), ((0.5, 1), (1.0, 2), (1.5, 1)))
            self.assertEqual(queue.events_for(second), queue.events_for(first))
            self.assertEqual(queue.submit(second), "ready")
            self.assertEqual(len(client.calls), 1)

    def test_failed_analysis_adds_no_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "song.wav"
            source.write_bytes(b"audio")
            cases = [
                BeatAnalysisResult(status=STATUS_ERROR, message="tracker failed"),
                BeatAnalysisResult(status=STATUS_UNAVAILABLE, message="Madmom worker is not installed"),
                BeatAnalysisResult(status=STATUS_EMPTY),
                BeatAnalysisResult(status=STATUS_CANCELLED, message="analysis cancelled"),
            ]
            for result in cases:
                with self.subTest(status=result.status):
                    queue = BeatJobQueue(_Client(result=result))
                    self.assertEqual(queue.submit(source), "queued")
                    queue.drain()
                    self.assertEqual(queue.events_for(source), ())
                    finished = None
                    while True:
                        message = queue.poll()
                        if message is None:
                            break
                        if message[0] == "finished":
                            finished = message
                    self.assertIsNotNone(finished)
                    self.assertEqual(finished[2].status, result.status)
                    self.assertEqual(finished[2].events, result.events)
