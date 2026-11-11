"""Stem Splitter: models, devices, the panel and adding stems to the project.

Demucs is replaced by a small script with the same command line, so these
tests need neither PyTorch nor a download.
"""

import json
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SONICLAB_DEMUCS_PYTHON", "")

import numpy as np
import soundfile as sf
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from soniclab.core.preview import PreviewPlayer
from soniclab.core.stems import (
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_MPS,
    MODEL,
    MODELS,
    SIX_STEM_MODEL,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    STEMS,
    StemJobQueue,
    StemSplitter,
    existing_stems,
    export_stems,
    model_stems,
    probe_engine,
    split_folder,
    track_order,
)
from soniclab.ui.main_window import MainWindow
from soniclab.ui.stem_splitter import StemSplitterPanel

RATE = 44100
SIX = ("vocals", "drums", "bass", "guitar", "piano", "other")

# Stands in for the Demucs environment's Python. ``-m demucs ...`` writes the
# stems of the requested model; ``-c <probe> ...`` answers probe_engine.
FAKE_ENGINE = textwrap.dedent(
    """
    import json, os, pathlib, sys, time
    log = os.environ.get("FAKE_DEMUCS_LOG")
    if sys.argv[1] == "-c":
        answer = json.loads(os.environ.get("FAKE_PROBE", "{}"))
        print("SONICLAB_PROBE " + json.dumps(answer))
        sys.exit(0)
    import numpy as np, soundfile as sf
    args = sys.argv[3:]
    out = args[args.index("-o") + 1]
    model = args[args.index("-n") + 1]
    device = args[args.index("-d") + 1] if "-d" in args else "auto"
    track = args[-1]
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps({"model": model, "device": device, "args": args,
                                     "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")}) + "\\n")
    delay = float(os.environ.get("FAKE_DEMUCS_DELAY", "0"))
    stems = {"htdemucs_6s": ("drums", "bass", "other", "vocals", "guitar", "piano")}.get(
        model, ("drums", "bass", "other", "vocals"))
    audio, rate = sf.read(track, dtype="float32", always_2d=True)
    folder = pathlib.Path(out) / model
    folder.mkdir(parents=True, exist_ok=True)
    for percent in (0, 50, 100):
        sys.stderr.write(f"\\r{percent:3d}%|####| 1/2 [00:00<00:00]")
        sys.stderr.flush()
        if percent == 50:
            # Half the stems exist while the run is still going.
            for stem in stems[:2]:
                sf.write(folder / f"{stem}.wav", audio * 0.1, rate)
        time.sleep(delay)
    if device in os.environ.get("FAKE_DEMUCS_FAIL_DEVICES", "").split(","):
        sys.stderr.write("\\nRuntimeError: this device does not work\\n")
        sys.exit(1)
    for index, stem in enumerate(stems):
        sf.write(folder / f"{stem}.wav", audio * (index + 1) * 0.1, rate)
    """
)


def fake_python(folder: Path) -> Path:
    script = folder / "fake_engine.py"
    script.write_text(FAKE_ENGINE, encoding="utf-8")
    launcher = folder / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


def song(folder: Path, name="song.wav", seconds=1.0) -> Path:
    path = folder / name
    t = np.arange(int(RATE * seconds)) / RATE
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([tone, tone], axis=1), RATE)
    return path


class Environment(unittest.TestCase):
    """A temp folder, a fake Demucs and a song; fake settings are env vars."""

    VARIABLES = ("FAKE_DEMUCS_DELAY", "FAKE_DEMUCS_LOG", "FAKE_DEMUCS_FAIL_DEVICES", "FAKE_PROBE")

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.python = fake_python(self.root)
        self.song = song(self.root)
        self.stems = self.root / "Stems"
        self.log = self.root / "runs.jsonl"
        os.environ["FAKE_DEMUCS_LOG"] = str(self.log)

    def tearDown(self):
        for name in self.VARIABLES:
            os.environ.pop(name, None)
        self.folder.cleanup()

    def runs(self) -> list[dict]:
        if not self.log.is_file():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


class ModelTests(Environment):
    def test_six_stem_model_writes_six_stems_in_its_own_folder(self):
        splitter = StemSplitter(self.python)
        seen = []
        result = splitter.split(self.song, self.stems, on_progress=seen.append, model=SIX_STEM_MODEL)
        self.assertEqual(result.status, STATUS_OK, result.message)
        self.assertEqual(tuple(result.stems), SIX)
        self.assertEqual(result.model, SIX_STEM_MODEL)
        target = split_folder(self.song, self.stems, SIX_STEM_MODEL)
        self.assertEqual(target.name, SIX_STEM_MODEL)
        for stem, path in result.stems.items():
            self.assertEqual(Path(path), target / f"{stem}.wav")
            # Every stem is as long as the song, so they line up in the DAW.
            self.assertEqual(sf.info(path).frames, sf.info(str(self.song)).frames)
        self.assertFalse((target / ".partial").exists())
        self.assertEqual(seen[-1], 1.0)
        self.assertTrue(splitter.split(self.song, self.stems, model=SIX_STEM_MODEL).cached)
        self.assertEqual(len(self.runs()), 1)

    def test_four_and_six_stem_sets_of_one_song_do_not_mix(self):
        splitter = StemSplitter(self.python)
        four = splitter.split(self.song, self.stems)
        six = splitter.split(self.song, self.stems, model=SIX_STEM_MODEL)
        self.assertEqual(tuple(four.stems), STEMS)
        self.assertEqual(Path(four.stems["drums"]).parent, split_folder(self.song, self.stems))
        self.assertEqual(Path(six.stems["drums"]).parent.parent, split_folder(self.song, self.stems))
        self.assertNotEqual(four.stems["drums"], six.stems["drums"])
        # A four-stem folder is never taken for a six-stem split.
        other = song(self.root, "other.wav")
        splitter.split(other, self.stems)
        self.assertIsNone(existing_stems(other, self.stems, SIX_STEM_MODEL))
        self.assertIsNotNone(existing_stems(other, self.stems, MODEL))

    def test_models_and_track_order(self):
        self.assertEqual(model_stems(SIX_STEM_MODEL), SIX)
        self.assertEqual(model_stems("unknown"), STEMS)
        self.assertNotIn("snare", {stem for model in MODELS.values() for stem in model.stems})
        note = MODELS[SIX_STEM_MODEL].note.lower()
        self.assertIn("piano", note)
        self.assertIn("snare stays inside drums", note)
        self.assertEqual(track_order(SIX), ("drums", "vocals", "bass", "guitar", "piano", "other"))
        self.assertEqual(track_order(("piano", "vocals")), ("piano", "vocals"))


class DeviceTests(Environment):
    def test_auto_passes_no_device_and_others_pass_theirs(self):
        splitter = StemSplitter(self.python)
        splitter.split(self.song, self.stems)
        splitter.split(song(self.root, "b.wav"), self.stems, device=DEVICE_CPU)
        splitter.split(song(self.root, "c.wav"), self.stems, device=DEVICE_CUDA, segment=4)
        runs = self.runs()
        self.assertEqual([run["device"] for run in runs], ["auto", "cpu", "cuda"])
        self.assertNotIn("-d", runs[0]["args"])
        self.assertEqual(runs[2]["args"][runs[2]["args"].index("--segment") + 1], "4")
        self.assertEqual([run["fallback"] for run in runs], ["", "", ""])

    def test_failed_apple_gpu_run_is_repeated_on_the_cpu(self):
        os.environ["FAKE_DEMUCS_FAIL_DEVICES"] = "mps"
        result = StemSplitter(self.python).split(self.song, self.stems, model=SIX_STEM_MODEL, device=DEVICE_MPS)
        self.assertEqual(result.status, STATUS_OK, result.message)
        self.assertEqual(result.device, DEVICE_CPU)
        self.assertIn("CPU was used", result.message)
        runs = self.runs()
        self.assertEqual([run["device"] for run in runs], ["mps", "cpu"])
        self.assertEqual([run["fallback"] for run in runs], ["1", ""])
        self.assertEqual(len(result.stems), 6)

    def test_working_apple_gpu_run_is_kept(self):
        result = StemSplitter(self.python).split(self.song, self.stems, device=DEVICE_MPS)
        self.assertEqual(result.device, DEVICE_MPS)
        self.assertEqual(result.message, "")
        self.assertEqual(len(self.runs()), 1)

    def test_failed_run_leaves_no_stems_behind(self):
        os.environ["FAKE_DEMUCS_FAIL_DEVICES"] = "cpu"
        result = StemSplitter(self.python).split(self.song, self.stems, model=SIX_STEM_MODEL, device=DEVICE_CPU)
        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("this device does not work", result.message)
        target = split_folder(self.song, self.stems, SIX_STEM_MODEL)
        self.assertEqual(list(target.glob("**/*.wav")), [])
        self.assertIsNone(existing_stems(self.song, self.stems, SIX_STEM_MODEL))

    def test_cancelled_run_leaves_no_stems_behind(self):
        os.environ["FAKE_DEMUCS_DELAY"] = "0.6"
        queue = StemJobQueue(StemSplitter(self.python))
        job = queue.submit(self.song, self.stems, model=SIX_STEM_MODEL)
        target = split_folder(self.song, self.stems, SIX_STEM_MODEL)
        deadline = time.monotonic() + 10
        # Wait until Demucs has written its first stems, then cancel.
        while time.monotonic() < deadline and not list(target.glob(".partial/**/*.wav")):
            time.sleep(0.02)
        self.assertTrue(list(target.glob(".partial/**/*.wav")))
        queue.cancel()
        queue.drain(5)
        finished = [message for message in iter(queue.poll, None) if message[0] == "finished"]
        self.assertEqual(finished[-1][1], job)
        self.assertEqual(finished[-1][2].status, STATUS_CANCELLED)
        self.assertEqual(list(target.glob("**/*.wav")), [])
        self.assertIsNone(existing_stems(self.song, self.stems, SIX_STEM_MODEL))


class EngineTests(Environment):
    def test_probe_reports_versions_devices_and_downloads(self):
        os.environ["FAKE_PROBE"] = json.dumps(
            {"torch": "2.2.2", "demucs": "4.0.1", "cuda": False, "mps": True,
             "weights": {SIX_STEM_MODEL: False, MODEL: True}}
        )
        info = probe_engine(self.python)
        self.assertTrue(info.available)
        self.assertEqual((info.demucs_version, info.torch_version), ("4.0.1", "2.2.2"))
        self.assertEqual(info.devices, ("auto", "cpu", "mps"))
        self.assertIs(info.downloaded[SIX_STEM_MODEL], False)
        self.assertIs(info.downloaded[MODEL], True)
        self.assertIsNone(info.downloaded["htdemucs"])

    def test_probe_of_a_python_without_demucs(self):
        info = probe_engine(sys.executable)
        self.assertFalse(info.available)
        self.assertIn("Demucs is not installed", info.message)
        self.assertFalse(probe_engine(self.root / "missing" / "python").available)

    def test_export_names_files_and_never_overwrites(self):
        result = StemSplitter(self.python).split(self.song, self.stems, model=SIX_STEM_MODEL)
        chosen = {stem: result.stems[stem] for stem in ("piano", "vocals")}
        out = self.root / "export"
        first = export_stems(chosen, out, "My Song")
        self.assertEqual(sorted(path.name for path in first), ["My Song - piano.wav", "My Song - vocals.wav"])
        second = export_stems(chosen, out, "My Song")
        self.assertEqual(sorted(path.name for path in second), ["My Song - piano (2).wav", "My Song - vocals (2).wav"])
        self.assertEqual(len(list(out.iterdir())), 4)
        audio, _rate = sf.read(str(out / "My Song - piano.wav"))
        source, _rate = sf.read(result.stems["piano"])
        self.assertTrue(np.array_equal(audio, source))


class FakeStream:
    def __init__(self, samplerate, channels, callback):
        self.samplerate, self.channels, self.callback = samplerate, channels, callback
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def pull(self, frames):
        out = np.full((frames, self.channels), 9.0, dtype=np.float32)
        self.callback(out, frames, None, None)
        return out


class PreviewTests(Environment):
    def test_plays_to_the_end_then_reports_finished(self):
        streams = []
        player = PreviewPlayer(lambda *args: streams.append(FakeStream(*args)) or streams[-1])
        self.assertEqual(player.play(self.song), "")
        self.assertTrue(player.playing)
        self.assertEqual(player.path, str(self.song))
        source, _rate = sf.read(str(self.song), dtype="float32", always_2d=True)
        first = streams[0].pull(1000)
        self.assertTrue(np.array_equal(first, source[:1000]))
        last = streams[0].pull(RATE)  # more than is left
        self.assertTrue(np.array_equal(last[: RATE - 1000], source[1000:]))
        self.assertTrue(np.all(last[RATE - 1000 :] == 0.0))
        self.assertFalse(player.playing)
        self.assertIsNone(player.path)

    def test_stop_missing_file_and_no_device(self):
        streams = []
        player = PreviewPlayer(lambda *args: streams.append(FakeStream(*args)) or streams[-1])
        player.play(self.song)
        player.stop()
        self.assertTrue(streams[0].closed)
        self.assertFalse(player.playing)
        self.assertIn("Could not read", player.play(self.root / "missing.wav"))

        def no_device(*_args):
            raise RuntimeError("no output device")

        broken = PreviewPlayer(no_device)
        self.assertIn("no output device", broken.play(self.song))
        self.assertFalse(broken.playing)


class PanelTests(Environment):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.streams = []
        player = PreviewPlayer(lambda *args: self.streams.append(FakeStream(*args)) or self.streams[-1])
        self.panel = StemSplitterPanel(StemJobQueue(StemSplitter(self.python)), player)
        self.panel.default_folder = lambda: self.stems

    def tearDown(self):
        self.panel.shutdown()
        self.panel.jobs.drain(5)
        self.panel.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def wait(self, condition, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not condition():
            self.panel.poll()
            self.app.processEvents()
            time.sleep(0.02)
        self.assertTrue(condition())

    def load(self, path=None):
        self.panel.set_source(path or self.song)
        self.wait(lambda: self.panel._loading is None)

    def test_split_six_stems_then_export_and_add(self):
        panel = self.panel
        self.assertEqual(panel.model, SIX_STEM_MODEL)
        self.assertFalse(panel.split_button.isEnabled())
        self.load()
        self.assertIn("0:01", panel.source_info.text())
        self.assertIn("44.1 kHz", panel.source_info.text())
        self.assertTrue(panel.split_button.isEnabled())
        self.assertFalse(panel.add_button.isEnabled())

        panel.start()
        self.assertTrue(panel.splitting)
        self.assertTrue(panel.cancel_button.isEnabled())
        self.assertFalse(panel.split_button.isEnabled())
        self.wait(lambda: not panel.splitting)
        self.assertEqual(tuple(panel.stems), SIX)
        self.assertIn("Split into 6 stems", panel.status_label.text())
        self.assertEqual(panel.progress.value(), 1000)
        self.assertTrue(all(box.isEnabled() and box.isChecked() for box, _play in panel._rows.values()))

        # Untick everything but piano and drums.
        for stem, (box, _play) in panel._rows.items():
            box.setChecked(stem in ("piano", "drums"))
        self.assertEqual(sorted(panel.selected_stems()), ["drums", "piano"])
        written = panel.export_to(self.root / "out")
        self.assertEqual(sorted(path.name for path in written), ["song - drums.wav", "song - piano.wav"])
        self.assertEqual(len(panel.export_to(self.root / "all", panel.stems)), 6)

        requests = []
        panel.add_requested.connect(lambda *args: requests.append(args))
        panel.add_selected()
        name, stems, order = requests[0]
        self.assertEqual(name, "song")
        self.assertEqual(order, ("drums", "piano"))
        self.assertEqual(sorted(stems), ["drums", "piano"])

    def test_cancel_saves_nothing_and_allows_another_try(self):
        os.environ["FAKE_DEMUCS_DELAY"] = "0.5"
        panel = self.panel
        self.load()
        panel.start()
        self.wait(lambda: panel.progress.value() > 0 or not panel.splitting)
        panel.cancel()
        self.wait(lambda: not panel.splitting)
        self.assertEqual(panel.stems, {})
        self.assertIn("cancelled", panel.status_label.text())
        self.assertFalse(panel.add_button.isEnabled())
        self.assertTrue(panel.split_button.isEnabled())
        os.environ["FAKE_DEMUCS_DELAY"] = "0"
        panel.start()
        self.wait(lambda: not panel.splitting)
        self.assertEqual(len(panel.stems), 6)

    def test_saved_stems_are_found_per_model(self):
        StemSplitter(self.python).split(self.song, self.stems, model=SIX_STEM_MODEL)
        self.load()
        self.assertEqual(len(self.panel.stems), 6)
        self.assertIn("Saved stems found", self.panel.status_label.text())
        self.assertTrue(self.panel.add_button.isEnabled())
        # The four-stem model has not been run for this song.
        self.panel.model_box.setCurrentIndex(self.panel.model_box.findData(MODEL))
        self.assertEqual(self.panel.stems, {})
        self.assertEqual(sorted(self.panel._rows), sorted(STEMS))
        self.assertFalse(self.panel.add_button.isEnabled())

    def test_missing_engine_and_failures_are_explained(self):
        panel = StemSplitterPanel(StemJobQueue(StemSplitter(self.root / "missing" / "python")), self.panel.player)
        panel.default_folder = lambda: self.stems
        panel.set_source(self.song)
        deadline = time.monotonic() + 10
        while panel._loading is not None and time.monotonic() < deadline:
            panel.poll()
            time.sleep(0.02)
        panel.start()
        while panel.splitting and time.monotonic() < deadline:
            panel.poll()
            time.sleep(0.02)
        self.assertIn("Demucs is not installed", panel.status_label.text())
        self.assertIn("docs/stem_splitter.md", panel.status_label.text())
        panel.check_engine()
        while panel._probing and time.monotonic() < deadline:
            panel.poll()
            time.sleep(0.02)
        self.assertIn("Demucs is not installed", panel.engine_label.text())
        panel.shutdown()
        panel.deleteLater()

        self.panel.set_source(self.root / "not-audio.wav")
        self.wait(lambda: self.panel._loading is None)
        self.assertIsNone(self.panel.source)
        self.assertIn("Could not read", self.panel.source_info.text())

    def test_engine_probe_fills_the_device_list(self):
        os.environ["FAKE_PROBE"] = json.dumps(
            {"torch": "2.5.0", "demucs": "4.0.1", "cuda": False, "mps": True, "weights": {SIX_STEM_MODEL: False}}
        )
        self.panel.check_engine()
        self.wait(lambda: not self.panel._probing)
        devices = [self.panel.device_box.itemData(i) for i in range(self.panel.device_box.count())]
        self.assertEqual(devices, ["auto", "cpu", "mps"])
        self.assertEqual(self.panel.device, "auto")  # the Apple GPU is never chosen for the user
        self.assertIn("experimental", self.panel.device_box.itemText(2))
        self.assertIn("Demucs 4.0.1", self.panel.engine_label.text())
        self.assertIn("downloads on the first split", self.panel.engine_label.text())
        self.load()
        self.panel.device_box.setCurrentIndex(2)
        self.panel.start()
        self.wait(lambda: not self.panel.splitting)
        self.assertEqual(self.runs()[-1]["device"], "mps")

    def test_preview_plays_one_thing_at_a_time(self):
        panel = self.panel
        self.load()
        panel.start()
        self.wait(lambda: not panel.splitting)
        panel.preview("piano")
        self.assertEqual(panel.player.path, panel.stems["piano"])
        self.assertTrue(panel._rows["piano"][1].isChecked())
        panel.preview("vocals")
        self.assertEqual(panel.player.path, panel.stems["vocals"])
        self.assertFalse(panel._rows["piano"][1].isChecked())
        panel.preview(None)
        self.assertEqual(panel.player.path, str(self.song))
        self.assertTrue(panel.play_source.isChecked())
        panel.preview(None)  # the second click stops
        self.assertFalse(panel.player.playing)
        self.assertFalse(panel.play_source.isChecked())
        panel.preview("drums")
        panel.stop_preview()
        self.assertFalse(panel.player.playing)

    def test_busy_elsewhere_blocks_a_second_split(self):
        self.load()
        self.panel.busy_elsewhere = lambda: True
        self.panel.start()
        self.assertFalse(self.panel.splitting)
        self.assertIn("Another song is being split", self.panel.status_label.text())


class _Jobs:
    def __init__(self):
        self.submitted = []

    def events_for(self, path):
        return ()

    def result_for(self, path):
        return None

    def submit(self, path):
        self.submitted.append(os.path.basename(path))
        return "queued"

    def poll(self):
        return None

    def cancel(self):
        pass


class WindowTests(Environment):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.window = MainWindow()
        self.window.beat_jobs = self.jobs = _Jobs()
        self.window.current_path = str(self.root / "session.soniclab")
        self.window.show()
        self.result = StemSplitter(self.python).split(self.song, self.stems, model=SIX_STEM_MODEL)

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def test_panel_is_a_dock_that_saves_beside_the_project(self):
        window = self.window
        self.assertIs(window.stem_splitter_dock.widget(), window.stem_splitter)
        self.assertEqual(window.stem_splitter.folder, self.root / "Stems")
        window._show_stem_splitter()
        self.assertTrue(window.stem_splitter_dock.isVisible())
        labels = [action.text() for action in window.findChildren(QAction)]
        self.assertIn("Stem Splitter…", labels)

    def test_added_stems_become_aligned_tracks_led_by_drums(self):
        window = self.window
        order = track_order(SIX)
        window.stem_splitter.add_requested.emit("My Song", dict(self.result.stems), order)
        names = [track["name"] for track in window.project.tracks]
        self.assertEqual(names, [f"My Song {stem}" for stem in order])
        self.assertTrue(window.project.tracks[0]["drum_reference"])
        self.assertTrue(all(track["start"] == 0.0 for track in window.project.tracks))
        durations = {round(track["duration"], 6) for track in window.project.tracks}
        self.assertEqual(durations, {1.0})
        self.assertEqual(window.engine.clip_count, 6)
        self.assertEqual(self.jobs.submitted, [f"{stem}.wav" for stem in order])
        self.assertIn("Added 6 stems of “My Song”", window.statusBar().currentMessage())
        window._undo()
        self.assertEqual(window.project.tracks, [])

    def test_adding_only_piano_does_not_call_it_the_drum_track(self):
        window = self.window
        window.stem_splitter.add_requested.emit("My Song", {"piano": self.result.stems["piano"]}, ("piano",))
        self.assertEqual([track["name"] for track in window.project.tracks], ["My Song piano"])
        self.assertFalse(window.project.tracks[0].get("drum_reference"))
        self.assertIn("Added 1 stem of", window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
