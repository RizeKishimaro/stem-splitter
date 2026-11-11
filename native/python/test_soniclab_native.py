"""Python ↔ native prototype tests. Skipped when libsoniclab_engine is not built.

Run from the repo root after building native/ (see native/README.md):
    .venv/bin/python -m unittest discover -s native/python -v
"""

import gc
import time
import unittest
from pathlib import Path

import numpy as np

import soniclab_native
from soniclab.core.audio_engine import render_clips

try:
    soniclab_native.load_library()
    _MISSING = None
except OSError as exc:
    _MISSING = str(exc)

REPO = Path(__file__).resolve().parents[2]


def _python_master(audio, gain, muted):
    """The Python engine's own master stage on one centred, unity clip."""
    clip = {"audio": audio, "start": 0, "gain": 1.0, "pan": 0.0, "muted": False}
    return render_clips([clip], 0, len(audio), output_gain=gain, master_muted=muted)


@unittest.skipIf(_MISSING, _MISSING)
class NativeEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = soniclab_native.NativeEngine(44100.0, 2, 512)

    def tearDown(self):
        self.engine.close()

    def test_process_matches_python_master_stage(self):
        rng = np.random.default_rng(7)
        audio = rng.uniform(-1.0, 1.0, size=(512, 2)).astype(np.float32)
        for gain, muted in ((1.0, False), (0.85, False), (1.5, False), (0.0, False), (1.2, True)):
            self.engine.set_output_gain(gain)
            self.engine.set_master_muted(muted)
            native = self.engine.process(audio.copy())
            np.testing.assert_array_equal(native, _python_master(audio, gain, muted))

    def test_process_works_in_place_on_the_borrowed_array(self):
        buffer = np.full((4, 2), 0.5, dtype=np.float32)
        self.engine.set_output_gain(0.5)
        result = self.engine.process(buffer)
        self.assertIs(result, buffer)
        self.assertTrue(np.all(buffer == 0.25))
        self.assertAlmostEqual(self.engine.take_peak(), 0.25)

    def test_borrowed_buffer_can_be_freed_after_the_call(self):
        # Native code keeps no pointer: freeing the array and processing
        # fresh ones must not touch freed memory (ASan build catches it).
        for _ in range(50):
            buffer = np.ones((256, 2), dtype=np.float32)
            self.engine.process(buffer)
            del buffer
            gc.collect()
        self.assertEqual(self.engine.stats()["blocks_processed"], 50)

    def test_process_rejects_buffers_it_cannot_borrow_safely(self):
        with self.assertRaises(TypeError):
            self.engine.process(np.zeros((4, 2), dtype=np.float64))
        with self.assertRaises(ValueError):
            self.engine.process(np.zeros((4, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.engine.process(np.zeros((2, 4), dtype=np.float32).T)  # not C-contiguous
        readonly = np.zeros((4, 2), dtype=np.float32)
        readonly.flags.writeable = False
        with self.assertRaises(ValueError):
            self.engine.process(readonly)
        with self.assertRaises(soniclab_native.NativeEngineError) as raised:
            self.engine.process(np.zeros((513, 2), dtype=np.float32))
        self.assertEqual(raised.exception.code, -2)

    def test_device_thread_keeps_running_while_python_holds_the_gil(self):
        self.engine.start_device(440.0)
        self.assertTrue(self.engine.device_running)
        time.sleep(0.05)
        before = self.engine.stats()["blocks_processed"]
        # Pure-Python busy loop: this thread holds the GIL throughout. The
        # native audio thread needs no GIL, so it keeps producing blocks.
        deadline = time.perf_counter() + 0.3
        spins = 0
        while time.perf_counter() < deadline:
            spins += 1
        produced = self.engine.stats()["blocks_processed"] - before
        expected = 0.3 * 44100 / 512  # about 26 blocks
        self.assertGreater(produced, expected * 0.6)
        self.assertGreater(spins, 0)

        with self.assertRaises(soniclab_native.NativeEngineError) as raised:
            self.engine.process(np.zeros((4, 2), dtype=np.float32))
        self.assertEqual(raised.exception.code, -3)

        out = self.engine.read_output(44100)
        self.assertGreater(len(out), 4096)
        self.assertEqual(out.shape[1], 2)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.5, places=2)
        self.engine.stop_device()
        self.assertFalse(self.engine.device_running)

    def test_gain_change_reaches_running_device(self):
        self.engine.start_device(220.0)
        self.engine.set_output_gain(0.5)
        time.sleep(0.05)
        self.engine.read_output(44100)  # discard audio from before the change
        self.engine.take_peak()
        time.sleep(0.1)
        self.assertAlmostEqual(self.engine.take_peak(), 0.25, places=2)
        self.assertGreaterEqual(self.engine.stats()["commands_applied"], 1)

    def test_close_is_idempotent_and_final(self):
        self.engine.start_device(440.0)
        self.engine.close()  # also stops the device thread
        self.engine.close()
        with self.assertRaises(ValueError):
            self.engine.take_peak()

    def test_context_manager_releases_handle(self):
        with soniclab_native.NativeEngine(48000.0, 1, 64) as engine:
            engine.process(np.zeros((64, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            engine.stats()


class PythonEngineStaysDefaultTests(unittest.TestCase):
    def test_app_does_not_import_the_prototype(self):
        users = [
            path.relative_to(REPO)
            for path in (REPO / "soniclab").rglob("*.py")
            if "soniclab_native" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(users, [])
