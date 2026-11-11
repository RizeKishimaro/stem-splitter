import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from soniclab.core.beat_analysis import (
    BEAT_POSITION_SEMANTICS,
    CACHE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    STATUS_CANCELLED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_UNAVAILABLE,
    TIME_UNIT,
    WORKER_VERSION,
    BeatAnalysisClient,
    BeatAnalysisError,
    cache_key,
    file_sha256,
    parse_worker_document,
)


BURMA_HOUSE = Path("/home/rizekishimaro/Music/Burma House [2HcLKKpY4Dk].mp3")
RUN_MADMOM = os.environ.get("SONICLAB_MADMOM_INTEGRATION") == "1"


def _worker_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "madmom_worker.py"
    spec = importlib.util.spec_from_file_location("madmom_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document(**overrides):
    document = {
        "schema_version": SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "engine": "madmom",
        "engine_version": "0.16.1",
        "status": STATUS_OK,
        "message": "",
        "time_unit": TIME_UNIT,
        "beat_position_semantics": BEAT_POSITION_SEMANTICS,
        "tempo_unit": "bpm",
        "tempo_bpm": 120.0,
        "fps": 100,
        "beats_per_bar": [3, 4],
        "sample_rate": None,
        "events": [
            {"time_seconds": 0.5, "beat_position": 1},
            {"time_seconds": 1.0, "beat_position": 2},
            {"time_seconds": 1.5, "beat_position": 3},
            {"time_seconds": 2.0, "beat_position": 4},
        ],
    }
    document.update(overrides)
    return document


def _client(root: Path, script: str) -> tuple[BeatAnalysisClient, Path]:
    worker = root / "worker.py"
    worker.write_text(script, encoding="utf-8")
    cache = root / "cache"
    return BeatAnalysisClient(python=sys.executable, worker_script=worker, cache_dir=cache, timeout=5), cache


def _emitting_script(root: Path, document: dict, *, preamble: str = "") -> str:
    payload = json.dumps(document)
    return f"""
import json, sys
from pathlib import Path
counter = Path({str(root / "count")!r})
count = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(count + 1))
Path({str(root / "argv")!r}).write_text(sys.argv[1])
{preamble}
print("stage detecting", file=sys.stderr, flush=True)
print("library warning", file=sys.stderr, flush=True)
print("stage tracking", file=sys.stderr, flush=True)
json.dump(json.loads({payload!r}), sys.stdout)
sys.stdout.write("\\n")
sys.stdout.flush()
"""


class DocumentTests(unittest.TestCase):
    def test_seconds_and_downbeat_position_are_explicit(self):
        parsed = parse_worker_document(_document(), fps=100, beats_per_bar=(3, 4))
        self.assertEqual(parsed["time_unit"], "seconds")
        self.assertEqual(parsed["beat_position_semantics"], "1_is_downbeat")
        self.assertEqual(parsed["events"][0]["beat_position"], 1)
        self.assertEqual(parsed["events"][0]["time_seconds"], 0.5)

    def test_worker_module_uses_the_same_contract(self):
        worker = _worker_module()
        self.assertEqual(worker.SCHEMA_VERSION, SCHEMA_VERSION)
        self.assertEqual(worker.WORKER_VERSION, WORKER_VERSION)
        self.assertEqual(worker.TIME_UNIT, TIME_UNIT)
        self.assertEqual(worker.BEAT_POSITION_SEMANTICS, BEAT_POSITION_SEMANTICS)
        events = worker._events([[0.48, 1], [0.96, 2], [1.44, 3], [1.92, 4]])
        self.assertEqual([event["beat_position"] for event in events], [1, 2, 3, 4])
        self.assertAlmostEqual(worker._tempo(events), 125.0, places=6)

    def test_rejects_wrong_units_positions_and_versions(self):
        cases = [
            {"time_unit": "samples"},
            {"beat_position_semantics": "index_mod_meter"},
            {"schema_version": 2},
            {"worker_version": "0"},
            {"tempo_unit": "hz"},
            {"events": [{"time_seconds": -0.1, "beat_position": 1}]},
            {"events": [{"time_seconds": 0.2, "beat_position": 0}], "status": STATUS_OK, "tempo_bpm": None},
            {"events": [{"time_seconds": 0.2, "beat_position": True}], "status": STATUS_OK, "tempo_bpm": None},
            {"events": [{"time_seconds": 1.0, "beat_position": 5}], "status": STATUS_OK, "tempo_bpm": None},
            {
                "events": [
                    {"time_seconds": 1.0, "beat_position": 1},
                    {"time_seconds": 1.0, "beat_position": 2},
                ]
            },
            {"tempo_bpm": float("nan")},
            {"status": STATUS_EMPTY, "events": [{"time_seconds": 0.2, "beat_position": 1}], "tempo_bpm": None},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(BeatAnalysisError):
                    parse_worker_document(_document(**overrides), fps=100, beats_per_bar=(3, 4))


class ClientTests(unittest.TestCase):
    def test_existing_path_is_not_copied_and_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.mp3"
            source.write_bytes(b"not-a-real-mpeg-stream")
            client, cache = _client(root, _emitting_script(root, _document()))
            stages = []
            created = {"count": 0}
            real_mkstemp = tempfile.mkstemp

            def spy_mkstemp(*args, **kwargs):
                created["count"] += 1
                return real_mkstemp(*args, **kwargs)

            original = tempfile.mkstemp
            tempfile.mkstemp = spy_mkstemp
            try:
                first = client.analyze_path(source, on_progress=stages.append)
                second = client.analyze(source)
            finally:
                tempfile.mkstemp = original
            self.assertEqual(created["count"], 0)
            self.assertEqual(Path((root / "argv").read_text()), source.resolve())
            self.assertEqual(first.status, STATUS_OK)
            self.assertFalse(first.cached)
            self.assertEqual(stages, ["detecting", "tracking"])
            self.assertIn("library warning", first.stderr)
            self.assertEqual(first.events[0].time_seconds, 0.5)
            self.assertTrue(first.events[0].is_downbeat)
            self.assertFalse(first.events[1].is_downbeat)
            self.assertEqual(len(first.downbeats), 1)
            self.assertTrue(second.cached)
            self.assertEqual(len(second.events), len(first.events))
            self.assertEqual((root / "count").read_text(), "1")
            self.assertEqual(len(list(cache.glob("*.json"))), 1)
            stored = json.loads(next(cache.glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(stored["cache_schema_version"], CACHE_SCHEMA_VERSION)
            self.assertEqual(stored["worker_version"], WORKER_VERSION)
            self.assertEqual(stored["content_sha256"], file_sha256(source.resolve()))

    def test_old_cache_schema_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.wav"
            source.write_bytes(b"RIFFstub")
            client, cache = _client(root, _emitting_script(root, _document()))
            content_hash = file_sha256(source.resolve())
            key = cache_key(content_hash, 100, (3, 4))
            cache.mkdir()
            (cache / f"{key}.json").write_text(
                json.dumps(
                    {
                        "cache_schema_version": CACHE_SCHEMA_VERSION + 9,
                        "worker_version": WORKER_VERSION,
                        "content_sha256": content_hash,
                        "fps": 100,
                        "beats_per_bar": [3, 4],
                        "result": _document(),
                    }
                ),
                encoding="utf-8",
            )
            result = client.analyze_path(source)
            self.assertFalse(result.cached)
            self.assertEqual((root / "count").read_text(), "1")
            refreshed = json.loads((cache / f"{key}.json").read_text(encoding="utf-8"))
            self.assertEqual(refreshed["cache_schema_version"], CACHE_SCHEMA_VERSION)

    def test_valid_json_with_nonzero_exit_is_not_accepted_or_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.wav"
            source.write_bytes(b"RIFFstub")
            script = _emitting_script(root, _document(message="tracker stopped")) + "\nraise SystemExit(2)\n"
            client, cache = _client(root, script)
            result = client.analyze_path(source)
            self.assertEqual(result.status, STATUS_ERROR)
            self.assertEqual(result.events, ())
            self.assertIn("tracker stopped", result.message)
            self.assertIn("worker exit 2", result.message)
            self.assertEqual(list(cache.glob("*.json")), [])
            again = client.analyze_path(source)
            self.assertEqual(again.status, STATUS_ERROR)
            self.assertFalse(again.cached)
            self.assertEqual((root / "count").read_text(), "2")

    def test_invalid_json_timeout_and_cancel_are_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.wav"
            source.write_bytes(b"RIFFstub")
            client, cache = _client(root, "import sys\nprint('not-json', flush=True)\nprint('boom', file=sys.stderr, flush=True)\n")
            invalid = client.analyze_path(source)
            self.assertEqual(invalid.status, STATUS_ERROR)
            self.assertIn("boom", invalid.stderr)
            self.assertEqual(list(cache.glob("*.json")), [])

            sleeper = "import time\ntime.sleep(30)\n"
            client, cache = _client(root, sleeper)
            client.timeout = 0.3
            timed_out = client.analyze_path(source)
            self.assertEqual(timed_out.status, STATUS_TIMEOUT)
            self.assertEqual(list(cache.glob("*.json")), [])

            client, cache = _client(root, sleeper)
            cancel = threading.Event()

            def run():
                holder.append(client.analyze_path(source, cancel_event=cancel, timeout=10))

            holder = []
            thread = threading.Thread(target=run)
            thread.start()
            cancel.set()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(holder[0].status, STATUS_CANCELLED)
            self.assertEqual(list(cache.glob("*.json")), [])

    def test_missing_worker_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.wav"
            source.write_bytes(b"RIFFstub")
            client = BeatAnalysisClient(
                python=root / "missing-python",
                worker_script=root / "missing-worker.py",
                cache_dir=root / "cache",
            )
            result = client.analyze_path(source)
            self.assertEqual(result.status, STATUS_UNAVAILABLE)
            self.assertEqual(result.events, ())

    def test_memory_audio_uses_a_temporary_wav_only_on_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen = []
            client, _cache = _client(root, _emitting_script(root, _document()))
            calls = {"count": 0}
            real_mkstemp = tempfile.mkstemp

            def spy_mkstemp(*args, **kwargs):
                calls["count"] += 1
                descriptor, name = real_mkstemp(*args, **kwargs)
                seen.append(name)
                return descriptor, name

            original = tempfile.mkstemp
            tempfile.mkstemp = spy_mkstemp
            try:
                audio = np.zeros((32, 2), dtype=np.float32)
                first = client.analyze(audio, 44100)
                second = client.analyze(audio, 44100)
            finally:
                tempfile.mkstemp = original
            self.assertEqual(first.status, STATUS_OK)
            self.assertEqual(calls["count"], 1)
            self.assertTrue(seen[0].endswith(".wav"))
            self.assertFalse(Path(seen[0]).exists())
            self.assertTrue(second.cached)
            self.assertEqual(first.sample_rate, 44100)


@unittest.skipUnless(RUN_MADMOM, "set SONICLAB_MADMOM_INTEGRATION=1 to run Madmom")
class MadmomIntegrationTests(unittest.TestCase):
    def test_burma_house_downbeats(self):
        self.assertTrue(BURMA_HOUSE.is_file(), BURMA_HOUSE)
        with tempfile.TemporaryDirectory() as directory:
            client = BeatAnalysisClient(cache_dir=Path(directory), timeout=600)
            started = os.times()
            result = client.analyze_path(BURMA_HOUSE, fps=100, beats_per_bar=(3, 4))
            self.assertEqual(result.status, STATUS_OK, result.message)
            self.assertEqual(result.time_unit, "seconds")
            self.assertEqual(result.beat_position_semantics, "1_is_downbeat")
            self.assertEqual(result.engine, "madmom")
            self.assertFalse(result.cached)
            self.assertGreaterEqual(len(result.events), 240)
            self.assertLessEqual(len(result.events), 290)
            self.assertGreaterEqual(len(result.downbeats), 55)
            self.assertLessEqual(len(result.downbeats), 80)
            self.assertGreater(result.tempo_bpm, 120.0)
            self.assertLess(result.tempo_bpm, 130.0)
            times = [event.time_seconds for event in result.events]
            self.assertEqual(times, sorted(set(times)))
            self.assertTrue(all(event.beat_position in (1, 2, 3, 4) for event in result.events))
            again = client.analyze(BURMA_HOUSE)
            self.assertTrue(again.cached)
            self.assertEqual(len(again.events), len(result.events))
            self.assertLess(os.times().elapsed - started.elapsed, 600)
