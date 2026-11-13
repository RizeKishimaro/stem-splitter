import gc
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsLineItem, QMessageBox
from shiboken6 import isValid

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.project import Project
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import (
    HEADER_WIDTH,
    PIXELS_PER_SECOND,
    ClipItem,
    TimelineView,
    parse_time_signature,
)


def beat_seconds(bpm, signature):
    _beats, beat_unit = parse_time_signature(signature)
    return (4.0 / beat_unit) * 60.0 / max(1.0, float(bpm))


def bar_pixels(bpm, signature):
    beats, _unit = parse_time_signature(signature)
    return beats * beat_seconds(bpm, signature) * PIXELS_PER_SECOND


class ParseTimeSignatureTests(unittest.TestCase):
    def test_function_is_module_level(self):
        self.assertEqual(parse_time_signature.__qualname__, "parse_time_signature")

    def test_accepted_forms(self):
        self.assertEqual(parse_time_signature("4/4"), (4, 4))
        self.assertEqual(parse_time_signature("3/4"), (3, 4))
        self.assertEqual(parse_time_signature("6/8"), (6, 8))
        self.assertEqual(parse_time_signature(" 7 / 8 "), (7, 8))
        self.assertEqual(parse_time_signature((5, 4)), (5, 4))
        self.assertEqual(parse_time_signature([3, 4]), (3, 4))
        self.assertEqual(parse_time_signature([6]), (6, 4))
        self.assertEqual(parse_time_signature(5), (5, 4))
        self.assertEqual(parse_time_signature(6.0), (6, 4))

    def test_invalid_values_fall_back(self):
        for value in ("", "nope", "0/4", "4/0", None, [], [0], object()):
            self.assertEqual(parse_time_signature(value), (4, 4))


class AudioEngineTests(unittest.TestCase):
    def test_multitrack_mix_mute_and_seek(self):
        engine = AudioEngine()
        engine.add_clip(np.full((OUTPUT_RATE, 2), 0.25, dtype=np.float32), 0.0)
        engine.add_clip(np.full((OUTPUT_RATE, 1), 0.25, dtype=np.float32), 0.5)
        engine._playing = True

        early = np.zeros((128, 2), dtype=np.float32)
        engine._callback(early, 128, None, None)
        self.assertTrue(np.allclose(early, 0.25))

        engine.seek(0.5)
        overlapped = np.zeros((128, 2), dtype=np.float32)
        engine._callback(overlapped, 128, None, None)
        self.assertTrue(np.allclose(overlapped, 0.5))
        self.assertAlmostEqual(engine.position, 0.5 + 128 / OUTPUT_RATE, places=6)

        engine._clips[0]["muted"] = True
        muted = np.zeros((128, 2), dtype=np.float32)
        engine._callback(muted, 128, None, None)
        self.assertTrue(np.allclose(muted, 0.25))

        engine._playing = False
        silent = np.ones((32, 2), dtype=np.float32)
        position = engine.position
        engine._callback(silent, 32, None, None)
        self.assertTrue(np.allclose(silent, 0))
        self.assertAlmostEqual(engine.position, position)

        engine.stop()
        self.assertEqual(engine.position, 0)
        self.assertFalse(engine.playing)
        engine.close()

    def test_pan_and_solo_are_applied_in_the_mix(self):
        engine = AudioEngine()
        engine.add_clip(np.full((OUTPUT_RATE, 2), 0.5, dtype=np.float32), 0.0)
        engine.add_clip(np.full((OUTPUT_RATE, 2), 0.5, dtype=np.float32), 0.0)
        engine.set_clip_pan(0, 1.0)
        engine.set_clip_solo(0, True)
        engine._playing = True
        out = np.zeros((32, 2), dtype=np.float32)
        engine._callback(out, 32, None, None)
        self.assertTrue(np.allclose(out[:, 0], 0.0))
        self.assertTrue(np.allclose(out[:, 1], 0.5))
        engine.set_clip_solo(0, False)
        engine.set_clip_muted(0, True)
        both = np.zeros((32, 2), dtype=np.float32)
        engine._callback(both, 32, None, None)
        self.assertTrue(np.allclose(both, 0.5))
        engine.close()

    def test_sum_is_clipped_and_mono_is_stereo(self):
        engine = AudioEngine()
        engine.add_clip(np.full(64, 0.8, dtype=np.float32), 0.0)
        engine.add_clip(np.full((64, 2), 0.8, dtype=np.float32), 0.0)
        engine._playing = True
        out = np.zeros((64, 2), dtype=np.float32)
        engine._callback(out, 64, None, None)
        self.assertTrue(np.allclose(out, 1.0))
        engine.close()

    def test_resample_matches_output_rate(self):
        source = np.zeros((48000, 2), dtype=np.float32)
        converted = AudioEngine.resample(source, 48000)
        self.assertLess(abs(len(converted) - OUTPUT_RATE), 20)
        same = AudioEngine.resample(source[:1000], OUTPUT_RATE)
        self.assertEqual(len(same), 1000)

    def test_project_roundtrip_omits_runtime_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "song.soniclab")
            project = Project()
            project.bpm = 96
            project.time_signature = [6, 8]
            project.tracks = [
                {
                    "name": "Kick",
                    "path": "/tmp/kick.wav",
                    "start": 1.25,
                    "duration": 2.0,
                }
            ]
            project.save(path)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn("_audio", text)
            loaded = Project.load(path)
            self.assertEqual(loaded.bpm, 96)
            self.assertEqual(loaded.time_signature, [6, 8])
            self.assertEqual(loaded.tracks[0]["start"], 1.25)
            self.assertNotIn("_audio", loaded.tracks[0])


class TimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = TimelineView()
        self.moves = []

    def tearDown(self):
        self.view.clear_scene()
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def _track(self, start, amplitude, name="Audio"):
        audio = np.full((OUTPUT_RATE, 2), amplitude, dtype=np.float32)
        return {
            "name": name,
            "start": start,
            "_audio": audio,
            "duration": len(audio) / OUTPUT_RATE,
        }

    def _clips(self):
        return {
            item.clip_index: item
            for item in self.view.scene_obj.items()
            if isinstance(item, ClipItem)
        }

    def _lines_at(self, y1):
        found = []
        for item in self.view.scene_obj.items():
            if not isinstance(item, QGraphicsLineItem) or item.parentItem() is not None:
                continue
            line = item.line()
            if abs(line.y1() - y1) < 0.1 and abs(line.x2() - line.x1()) < 0.1:
                found.append(round(line.x1(), 2))
        return sorted(set(found))

    def test_rebuild_keeps_start_time_and_draws_waveform(self):
        tracks = [self._track(0.2, 1.0, "Loud"), self._track(1.0, 0.05, "Quiet")]

        def on_move(index, seconds):
            self.moves.append((index, seconds))
            tracks[index]["start"] = seconds

        self.view.rebuild(tracks, on_move, 120, "4/4")
        gc.collect()
        self.assertEqual(self.moves, [])
        self.assertAlmostEqual(tracks[0]["start"], 0.2)
        clips = self._clips()
        self.assertEqual(set(clips), {0, 1})
        self.assertAlmostEqual(
            clips[0].pos().x(),
            HEADER_WIDTH + 0.2 * PIXELS_PER_SECOND,
            places=4,
        )
        self.assertAlmostEqual(clips[0].pos().y(), 42)
        self.assertAlmostEqual(clips[1].pos().y(), 42 + 76)
        def wave_peak(clip):
            lows, highs = clip.envelope
            return float(np.maximum(np.abs(lows), np.abs(highs)).max())

        self.assertGreater(clips[0].envelope[0].size, 50)
        self.assertGreater(wave_peak(clips[0]), wave_peak(clips[1]))
        self.assertGreater(wave_peak(clips[1]), 0)
        self.assertFalse(
            any(isinstance(item, QGraphicsLineItem) for item in clips[0].childItems())
        )

    def test_drag_snaps_to_beats_and_stays_on_its_track(self):
        tracks = [self._track(0.0, 0.4), self._track(0.0, 0.4, "Two")]

        def on_move(index, seconds):
            self.moves.append((index, seconds))
            tracks[index]["start"] = seconds

        self.view.rebuild(tracks, on_move, 120, "4/4")
        clip = self._clips()[1]
        y = clip.pos().y()
        step = beat_seconds(120, "4/4") * PIXELS_PER_SECOND
        clip.setPos(HEADER_WIDTH + step * 0.7, y + 50)
        self.assertAlmostEqual(clip.pos().x(), HEADER_WIDTH + step)
        self.assertAlmostEqual(clip.pos().y(), y)
        self.assertAlmostEqual(tracks[1]["start"], beat_seconds(120, "4/4"))
        self.assertEqual(tracks[0]["start"], 0.0)
        self.assertEqual(self.moves[-1][0], 1)

    def test_time_signature_changes_grid_and_snap(self):
        tracks = [self._track(0.0, 0.5)]
        self.view.rebuild(tracks, lambda *_args: None, 120, (4, 4))
        gc.collect()
        four = self._lines_at(0)
        self.assertAlmostEqual(four[1] - four[0], bar_pixels(120, "4/4"), places=2)
        first_bar = [
            x
            for x in self._lines_at(32)
            if HEADER_WIDTH < x < HEADER_WIDTH + bar_pixels(120, "4/4") - 0.1
        ]
        self.assertEqual(len(first_bar), 3)

        self.view.rebuild(tracks, lambda *_args: None, 120, "6/8")
        six = self._lines_at(0)
        self.assertAlmostEqual(six[1] - six[0], bar_pixels(120, "6/8"), places=2)
        clip = self._clips()[0]
        moved = []
        clip.moved_callback = lambda index, seconds: moved.append(seconds)
        clip.setPos(HEADER_WIDTH + 20, clip.pos().y())
        self.assertAlmostEqual(clip.pos().x(), HEADER_WIDTH + 22.5, places=4)
        self.assertAlmostEqual(moved[-1], 0.25, places=4)

        self.view.rebuild(tracks, lambda *_args: None, 60, "bad")
        halved = self._lines_at(0)
        self.assertAlmostEqual(halved[1] - halved[0], bar_pixels(60, "4/4"), places=2)

    def test_seek_playhead_and_deleted_playhead(self):
        sought = []
        self.view.seek_requested.connect(sought.append)
        self.view.rebuild([self._track(0.0, 0.2)], lambda *_args: None, 120, "4/4")
        self.view.show()
        self.app.processEvents()

        ruler = self.view.mapFromScene(QPointF(HEADER_WIDTH + 90, 8))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            ruler,
        )
        self.assertTrue(sought)
        self.assertAlmostEqual(sought[-1], 1.0, places=3)

        self.view.set_playhead(sought[-1])
        playhead = self.view.playhead
        self.assertIsNotNone(playhead)
        self.assertAlmostEqual(playhead.line().x1(), HEADER_WIDTH + PIXELS_PER_SECOND)

        clip = self._clips()[0]
        before = len(sought)
        clip_pos = self.view.mapFromScene(clip.scenePos() + QPointF(12, 12))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            clip_pos,
        )
        self.assertEqual(len(sought), before)

        stale = self.view.playhead
        self.view.clear_scene()
        self.assertFalse(isValid(stale))
        self.view.set_playhead(2.0)
        self.assertTrue(isValid(self.view.playhead))
        self.assertIsNot(self.view.playhead, stale)
        self.assertAlmostEqual(
            self.view.playhead.line().x1(),
            HEADER_WIDTH + 2 * PIXELS_PER_SECOND,
        )

    def test_playback_keeps_the_playhead_in_the_middle(self):
        tracks = [self._track(0.0, 0.2, "Long")]
        tracks[0]["duration"] = 40.0
        self.view.resize(200, 180)
        self.view.rebuild(tracks, lambda *_args: None, 120, "4/4")
        self.view.show()
        self.app.processEvents()
        self.view.set_playhead(10.0, follow=True)
        self.app.processEvents()
        playhead_x = self.view.playhead.line().x1()
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        self.assertAlmostEqual(playhead_x, visible.center().x(), delta=2)
        self.view.horizontalScrollBar().setValue(0)
        self.app.processEvents()
        self.view.set_playhead(10.0, follow=False)
        self.app.processEvents()
        self.assertLess(self.view.horizontalScrollBar().value(), 20)

    def test_shift_wheel_scrolls_horizontally(self):
        tracks = [self._track(index * 2.0, 0.2, f"T{index}") for index in range(8)]
        self.view.resize(420, 220)
        self.view.rebuild(tracks, lambda *_args: None, 120, "4/4")
        self.view.show()
        self.app.processEvents()
        horizontal = self.view.horizontalScrollBar()
        vertical = self.view.verticalScrollBar()
        horizontal.setValue(300)
        vertical.setValue(40)
        self.app.processEvents()

        def wheel(delta_y, modifiers):
            position = QPointF(30, 30)
            event = QWheelEvent(
                position,
                self.view.mapToGlobal(position),
                QPoint(0, 0),
                QPoint(0, delta_y),
                Qt.MouseButton.NoButton,
                modifiers,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(self.view.viewport(), event)
            self.app.processEvents()

        wheel(-120, Qt.KeyboardModifier.ShiftModifier)
        self.assertGreater(horizontal.value(), 300)
        self.assertEqual(vertical.value(), 40)

        shifted = horizontal.value()
        wheel(120, Qt.KeyboardModifier.NoModifier)
        self.assertEqual(horizontal.value(), shifted)
        self.assertLess(vertical.value(), 40)

    def test_dragging_one_block_does_not_move_the_track(self):
        audio = np.full((OUTPUT_RATE * 4, 2), 0.4, dtype=np.float32)
        tracks = [{"name": "Song", "start": 0.0, "_audio": audio, "duration": 4.0}]
        moved = []
        self.view.on_block_move = lambda *args: moved.append(args)
        self.view.resize(900, 240)
        self.view.rebuild(tracks, lambda index, seconds: moved.append(("track", index, seconds)), 120, "4/4")
        self.view.show()
        self.app.processEvents()
        clip = self._clips()[0]
        self.assertGreater(len(clip._block_spans), 1)
        origin_x = clip.pos().x()
        local = QPointF(clip.pos().x() + 200, clip.pos().y() + 40)
        view_pt = self.view.mapFromScene(local)

        def send(kind, point, buttons):
            # Ctrl (⌘ on macOS) takes one bar; a plain drag moves the track.
            event = QMouseEvent(
                kind,
                QPointF(point),
                QPointF(self.view.viewport().mapToGlobal(point)),
                Qt.MouseButton.LeftButton,
                buttons,
                Qt.KeyboardModifier.ControlModifier,
            )
            QApplication.sendEvent(self.view.viewport(), event)
            self.app.processEvents()

        send(QEvent.Type.MouseButtonPress, view_pt, Qt.MouseButton.LeftButton)
        send(QEvent.Type.MouseMove, view_pt + QPoint(180, 0), Qt.MouseButton.LeftButton)
        send(QEvent.Type.MouseButtonRelease, view_pt + QPoint(180, 0), Qt.MouseButton.NoButton)
        self.app.processEvents()
        self.assertAlmostEqual(clip.pos().x(), origin_x)
        self.assertEqual(tracks[0]["start"], 0.0)
        self.assertFalse(any(item[0] == "track" for item in moved))
        self.assertTrue(moved)
        index, local_start, local_duration, new_start, share, _insert_at, bar, canvas, join = moved[-1]
        self.assertIsNone(canvas)
        self.assertFalse(join)  # same row: the bar stays on its own track
        self.assertEqual(index, 0)
        self.assertEqual(bar, 1)
        self.assertTrue(share)
        self.assertAlmostEqual(local_start, 2.0, places=3)
        self.assertAlmostEqual(local_duration, 2.0, places=3)
        self.assertAlmostEqual(new_start, 4.0, places=3)


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def _write_wav(self, name, amplitude, rate=OUTPUT_RATE):
        path = os.path.join(self.directory.name, name)
        audio = np.full((rate, 2), amplitude, dtype=np.float32)
        sf.write(path, audio, rate)
        return path

    def _import(self, paths):
        with patch(
            "soniclab.ui.main_window.QFileDialog.getOpenFileNames",
            return_value=(paths, ""),
        ):
            self.window._import_audio()
        self.window.finish_loading()  # files are decoded off the GUI thread

    def test_import_queues_the_source_for_beat_markers(self):
        class Jobs:
            def __init__(self):
                self.paths = []

            def submit(self, path):
                self.paths.append(os.path.abspath(path))
                return "queued"

            def poll(self):
                return None

            def events_for(self, _path):
                return ()

            def result_for(self, _path):
                return None

            def cancel(self):
                return None

        self.window.beat_jobs = Jobs()
        source = self._write_wav("beat-source.wav", 0.2)
        self._import([source])
        self.assertEqual(self.window.beat_jobs.paths, [os.path.abspath(source)])

    def test_import_mix_drag_tempo_and_seek(self):
        first = self._write_wav("a.wav", 0.4, 48000)
        second = self._write_wav("b.wav", 0.4)
        self._import([first, second])

        tracks = self.window.project.tracks
        self.assertEqual([track["start"] for track in tracks], [0.0, 0.0])
        self.assertAlmostEqual(tracks[0]["duration"], 1.0, places=2)
        self.assertEqual(len(self.window.engine._clips), 2)
        self.assertEqual(self.window.engine._clips[0]["start"], 0)
        self.assertLess(abs(len(tracks[0]["_audio"]) - OUTPUT_RATE), 20)

        # Resampling rings for a few samples, so mix a settled part of both clips.
        self.window.engine.seek(0.25)
        self.window.engine._playing = True
        mixed = np.zeros((64, 2), dtype=np.float32)
        self.window.engine._callback(mixed, 64, None, None)
        self.assertTrue(np.allclose(mixed, 0.8, atol=0.02))

        clips = {
            item.clip_index: item
            for item in self.window.timeline.scene_obj.items()
            if isinstance(item, ClipItem)
        }
        self.assertEqual(set(clips), {0, 1})
        self.assertGreater(clips[0].envelope[0].size, 50)
        clips[0].setPos(HEADER_WIDTH + 30, clips[0].pos().y() + 40)
        self.assertAlmostEqual(clips[0].pos().y(), 42)
        self.assertAlmostEqual(tracks[0]["start"], 0.5)
        self.assertEqual(self.window.engine._clips[0]["start"], int(0.5 * OUTPUT_RATE))
        self.assertEqual(tracks[1]["start"], 0.0)

        solo = np.zeros((64, 2), dtype=np.float32)
        self.window.engine.seek(0)
        self.window.engine._callback(solo, 64, None, None)
        self.assertTrue(np.allclose(solo, 0.4, atol=1e-3))

        self.window.bpm.setValue(90)
        self.assertEqual(tracks[0]["start"], 0.5)
        self.assertEqual(self.window.project.bpm, 90)
        self.window.signature.setCurrentText("3/4")
        self.assertEqual(self.window.project.time_signature, [3, 4])
        self.assertAlmostEqual(tracks[0]["start"], 0.5)
        self.assertAlmostEqual(
            self._clip_x(0),
            HEADER_WIDTH + 0.5 * PIXELS_PER_SECOND,
            places=3,
        )

        self.window.engine.seek(3)
        self.window._refresh_transport()
        self.assertEqual(self.window.position_label.text(), "00:03.000")
        self.assertAlmostEqual(
            self.window.timeline.playhead.line().x1(),
            HEADER_WIDTH + 3 * PIXELS_PER_SECOND,
        )
        self.window.timeline.clear_scene()
        self.window._refresh_transport()
        self.assertTrue(isValid(self.window.timeline.playhead))

        ruler = self.window.timeline.mapFromScene(QPointF(HEADER_WIDTH + 180, 6))
        QTest.mouseClick(
            self.window.timeline.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            ruler,
        )
        self.assertAlmostEqual(self.window.engine.position, 2.0, places=2)
        self.window._stop()
        self.assertEqual(self.window.engine.position, 0)

    def _clip_x(self, index):
        for item in self.window.timeline.scene_obj.items():
            if isinstance(item, ClipItem) and item.clip_index == index:
                return item.pos().x()
        self.fail(f"missing clip {index}")

    def test_channel_controls_reach_playback(self):
        path = self._write_wav("tone.wav", 0.4)
        self._import([path, path])
        self.window._set_pan(0, 1)
        self.window._set_solo(0, True)
        self.window.engine.seek(0.25)
        self.window.engine._playing = True
        out = np.zeros((32, 2), dtype=np.float32)
        self.window.engine._callback(out, 32, None, None)
        self.assertTrue(np.allclose(out[:, 0], 0.0, atol=1e-3))
        self.assertTrue(np.allclose(out[:, 1], 0.4, atol=1e-3))
        self.assertTrue(self.window.project.tracks[0]["solo"])
        self.assertFalse(self.window.project.tracks[1].get("solo", False))
        self.assertAlmostEqual(self.window.project.tracks[0]["pan"], 1.0)

    def test_failed_import_does_not_block_a_later_file(self):
        good = self._write_wav("good.wav", 0.2)
        bad = os.path.join(self.directory.name, "bad.wav")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("not audio")
        with patch.object(QMessageBox, "warning") as warning:
            self._import([bad, good])
        self.assertTrue(warning.called)
        self.assertEqual(len(self.window.project.tracks), 1)
        self.assertEqual(self.window.project.tracks[0]["start"], 0.0)
        self.assertTrue(
            any(isinstance(item, ClipItem) for item in self.window.timeline.scene_obj.items())
        )

    def test_save_and_open_restore_timeline_settings(self):
        source = self._write_wav("loop.wav", 0.3)
        self._import([source])
        clip = next(
            item
            for item in self.window.timeline.scene_obj.items()
            if isinstance(item, ClipItem)
        )
        clip.setPos(HEADER_WIDTH + 30, clip.pos().y())
        self.assertAlmostEqual(self.window.project.tracks[0]["start"], 0.5)
        self.window.bpm.setValue(140)
        self.window.signature.setCurrentText("7/8")
        self.assertAlmostEqual(self.window.project.tracks[0]["start"], 0.5)
        path = os.path.join(self.directory.name, "song.soniclab")
        self.window.current_path = path
        self.window._save_project()

        loaded = Project.load(path)
        self.assertEqual(loaded.bpm, 140)
        self.assertEqual(loaded.time_signature, [7, 8])
        with open(path, encoding="utf-8") as handle:
            self.assertNotIn("_audio", handle.read())
        self.assertAlmostEqual(loaded.tracks[0]["start"], 0.5)

        restored = MainWindow()
        try:
            with patch(
                "soniclab.ui.main_window.QFileDialog.getOpenFileName",
                return_value=(path, ""),
            ):
                restored._open_project()
            self.assertEqual(restored.bpm.value(), 140)
            self.assertEqual(restored.signature.currentText(), "7/8")
            self.assertEqual(restored.project.time_signature, [7, 8])
            self.assertAlmostEqual(restored.project.tracks[0]["start"], 0.5)
            self.assertEqual(
                restored.engine._clips[0]["start"],
                int(restored.project.tracks[0]["start"] * OUTPUT_RATE),
            )
            restored_clip = next(
                item
                for item in restored.timeline.scene_obj.items()
                if isinstance(item, ClipItem)
            )
            self.assertAlmostEqual(
                restored_clip.pos().x(),
                HEADER_WIDTH + restored.project.tracks[0]["start"] * PIXELS_PER_SECOND,
                places=3,
            )
            self.assertGreater(restored_clip.envelope[0].size, 50)
        finally:
            restored.close()

    def test_moved_block_stays_on_the_track_and_keeps_its_audio(self):
        audio = np.zeros((OUTPUT_RATE * 4, 2), dtype=np.float32)
        audio[: OUTPUT_RATE * 2] = 0.25
        audio[OUTPUT_RATE * 2 :] = 0.8
        self.window.project.tracks.append(
            {
                "name": "Song",
                "path": None,
                "start": 0.0,
                "duration": 4.0,
                "offset": 0.0,
                "_audio": audio,
            }
        )
        self.window._move_block(0, 2.0, 2.0, 4.0, True, 1, 1)
        tracks = self.window.project.tracks
        self.assertEqual(len(tracks), 2)
        self.assertAlmostEqual(tracks[0]["start"], 0.0)
        self.assertAlmostEqual(tracks[0]["duration"], 2.0)
        self.assertAlmostEqual(tracks[1]["start"], 4.0)
        self.assertAlmostEqual(tracks[1]["offset"], 2.0)
        self.assertAlmostEqual(tracks[1]["duration"], 2.0)
        self.assertEqual(tracks[0].get("lane"), tracks[1].get("lane"))
        early = self.window.engine.render(0, 64)
        gap = self.window.engine.render(2 * OUTPUT_RATE, 64)
        late = self.window.engine.render(4 * OUTPUT_RATE, 64)
        self.assertAlmostEqual(float(early[0, 0]), 0.25, places=3)
        self.assertAlmostEqual(float(gap[0, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(late[0, 0]), 0.8, places=3)
        clips = [
            item
            for item in self.window.timeline.scene_obj.items()
            if isinstance(item, ClipItem) and isValid(item)
        ]
        self.assertEqual(len(clips), 2)
        self.assertAlmostEqual(clips[0].pos().y(), clips[1].pos().y())


    def test_a_bar_block_moves_alone_again_and_again(self):
        # A left-drag takes the bar under the mouse; the track keeps its
        # other bars where they are. The moved bar is then one piece that
        # can be dragged again and again.
        audio = np.full((OUTPUT_RATE * 8, 2), 0.5, dtype=np.float32)
        window = self.window
        window.project.tracks.append(
            {"name": "drums", "path": None, "start": 0.0, "duration": 8.0, "offset": 0.0, "_audio": audio}
        )
        window.project.bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        window._reload_engine()
        window._refresh_timeline()
        view = window.timeline
        view.resize(1400, 300)
        window.show()
        self.app.processEvents()
        name = "drums"
        for expected in (1.0, 2.0, 3.0):
            clip = next(
                item for item in view.scene_obj.items()
                if isinstance(item, ClipItem) and isValid(item)
                and window.project.tracks[item.clip_index]["name"] == name
            )
            point = view.mapFromScene(QPointF(clip.pos().x() + 0.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75))
            target = point + QPoint(int(PIXELS_PER_SECOND), 0)
            for kind, at, buttons in (
                (QEvent.Type.MouseButtonPress, point, Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseMove, point + QPoint(int(PIXELS_PER_SECOND) // 2, 0), Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseMove, target, Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseButtonRelease, target, Qt.MouseButton.NoButton),
            ):
                event = QMouseEvent(kind, QPointF(at), QPointF(view.viewport().mapToGlobal(at)),
                                    Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
                QApplication.sendEvent(view.viewport(), event)
                self.app.processEvents()
            self.app.processEvents()
            name = "drums · bar 1"
            piece = next(track for track in window.project.tracks if track["name"] == name)
            self.assertAlmostEqual(piece["start"], expected, places=2)
            self.assertAlmostEqual(piece["duration"], 2.0)
            rest = next(track for track in window.project.tracks if track["name"] == "drums")
            self.assertAlmostEqual(rest["start"], 2.0)  # bars 2-4 never moved
            self.assertAlmostEqual(rest["duration"], 6.0)
        self.assertEqual(len(window.project.tracks), 2)

    def test_a_moved_bar_can_be_dragged_again_as_one_piece(self):
        # Regression: a bar dropped off the bar grid (here one beat late) was
        # cut in two at the next redraw, so a second drag tore half of it off
        # instead of moving it.
        audio = np.zeros((OUTPUT_RATE * 8, 2), dtype=np.float32)
        audio[OUTPUT_RATE * 2 : OUTPUT_RATE * 4] = 0.5
        window = self.window
        window.project.tracks.append(
            {"name": "drums", "path": None, "start": 0.0, "duration": 8.0, "offset": 0.0, "_audio": audio}
        )
        window.project.bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        window._reload_engine()
        window._refresh_timeline()
        window._move_block(0, 2.0, 2.0, 2.5, True, 1, 1)  # bar 2 lands one beat late, off the bar grid
        piece = next(track for track in window.project.tracks if track["name"] == "drums · bar 2")
        self.assertTrue(piece["bar_block"])
        view = window.timeline
        view.resize(1200, 300)
        window.show()
        self.app.processEvents()

        def piece_clip():
            index = window.project.tracks.index(piece)
            return next(
                item for item in view.scene_obj.items()
                if isinstance(item, ClipItem) and isValid(item) and item.clip_index == index
            )

        def drag(clip, dx):
            point = view.mapFromScene(QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.75))
            for kind, at, buttons in (
                (QEvent.Type.MouseButtonPress, point, Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseMove, point + QPoint(dx // 2, 0), Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseMove, point + QPoint(dx, 0), Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseButtonRelease, point + QPoint(dx, 0), Qt.MouseButton.NoButton),
            ):
                event = QMouseEvent(kind, QPointF(at), QPointF(view.viewport().mapToGlobal(at)),
                                    Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
                QApplication.sendEvent(view.viewport(), event)
                self.app.processEvents()
            self.app.processEvents()

        count = len(window.project.tracks)
        for expected in (3.5, 4.5):
            clip = piece_clip()
            self.assertEqual(len(clip._block_spans), 1)  # straddles a bar line, still one block
            self.assertTrue(clip.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            drag(clip, int(PIXELS_PER_SECOND))
            self.assertEqual(len(window.project.tracks), count)  # moved, not split again
            self.assertAlmostEqual(piece["start"], expected, places=2)
            self.assertAlmostEqual(piece["offset"], 2.0)
            self.assertAlmostEqual(piece["duration"], 2.0)
        # The whole bar plays at its new place.
        start = int(4.6 * OUTPUT_RATE)
        self.assertAlmostEqual(float(window.engine.render(start, 8)[0, 0]), 0.5, places=3)
        self.assertAlmostEqual(float(window.engine.render(int(6.4 * OUTPUT_RATE), 8)[0, 0]), 0.5, places=3)

        # Saved and reopened, the piece is still one block.
        window.project.tracks[:] = [dict(track, _audio=track["_audio"]) for track in window.project.tracks]
        window._refresh_timeline()
        self.assertEqual(len(piece_clip()._block_spans), 1)


def tearDownModule():
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()


if __name__ == "__main__":
    unittest.main()
