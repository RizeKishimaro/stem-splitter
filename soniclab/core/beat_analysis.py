"""Madmom beat analysis through an isolated Python 3.11 worker.

The worker is a separate process. This module never imports Madmom and never
runs inside the audio callback.

Time unit
    ``time_seconds`` is seconds from the start of the analyzed source, not
    arrangement time and not a sample index.

Beat position
    ``beat_position`` counts inside the bar the way musicians do. 1 is the
    downbeat. 2 is the next beat, and so on. A position is never inferred by
    dividing the event index by the session meter.

Cache
    Successful results are stored under a content hash. The file records
    ``cache_schema_version`` and ``worker_version``. A mismatch is a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
WORKER_VERSION = "1"
TIME_UNIT = "seconds"
BEAT_POSITION_SEMANTICS = "1_is_downbeat"
TEMPO_UNIT = "bpm"
ENGINE = "madmom"

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_UNAVAILABLE = "unavailable"
STATUS_CANCELLED = "cancelled"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

_CACHEABLE = {STATUS_OK, STATUS_EMPTY}
_STDERR_LIMIT = 16_384
_STDOUT_LIMIT = 2_000_000


@dataclass(frozen=True)
class BeatEvent:
    """One detected beat in the source audio."""

    time_seconds: float
    beat_position: int

    @property
    def is_downbeat(self) -> bool:
        return self.beat_position == 1


@dataclass(frozen=True)
class BeatAnalysisResult:
    """Validated analysis, or a failure that contains no invented beats."""

    status: str
    events: tuple[BeatEvent, ...] = ()
    tempo_bpm: float | None = None
    time_unit: str = TIME_UNIT
    beat_position_semantics: str = BEAT_POSITION_SEMANTICS
    tempo_unit: str = TEMPO_UNIT
    schema_version: int = SCHEMA_VERSION
    cache_schema_version: int = CACHE_SCHEMA_VERSION
    worker_version: str = WORKER_VERSION
    engine: str = ENGINE
    engine_version: str | None = None
    fps: int | None = None
    beats_per_bar: tuple[int, ...] = ()
    sample_rate: int | None = None
    message: str = ""
    stderr: str = ""
    cached: bool = False

    @property
    def downbeats(self) -> tuple[BeatEvent, ...]:
        return tuple(event for event in self.events if event.is_downbeat)


def _madmom_python_candidates() -> list[Path]:
    """Places a Madmom interpreter may live. The app's own Python is not one of them."""
    candidates = []
    override = os.environ.get("SONICLAB_MADMOM_PYTHON")
    if override:
        candidates.append(Path(override))
    installed = Path.home() / "Library" / "Application Support" / "SonicLab" / "madmom-venv"
    if (installed / ".soniclab-ready").is_file():
        candidates.append(installed / "bin" / "python")
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo / ".venv-madmom" / "bin" / "python")
    candidates.append(Path.cwd() / ".venv-madmom" / "bin" / "python")
    candidates.append(Path.home() / ".venvs" / "madmom" / "bin" / "python")
    return candidates


def default_madmom_python() -> Path | None:
    """Interpreter that has Madmom installed, if this machine has one.

    The application environment is not used. Madmom stays on Python 3.11
    because its code does not run on NumPy 2. A path that was set but does
    not exist is skipped so a stale setting cannot hide a working copy.
    """
    for candidate in _madmom_python_candidates():
        if candidate.is_file():
            return candidate
    return None


def default_worker_script() -> Path:
    override = os.environ.get("SONICLAB_MADMOM_WORKER")
    if override and Path(override).is_file():
        return Path(override)
    repo = Path(__file__).resolve().parents[2] / "tools" / "madmom_worker.py"
    if repo.is_file():
        return repo
    cwd = Path.cwd() / "tools" / "madmom_worker.py"
    if cwd.is_file():
        return cwd
    return Path.home() / "Library" / "Application Support" / "SonicLab" / "madmom_worker.py"


def _record_madmom(message: str) -> None:
    try:
        folder = Path.home() / ".cache" / "soniclab"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "diagnostics.log").open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except OSError:
        return


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "soniclab" / "beats"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def samples_sha256(audio, sample_rate: int) -> str:
    array = np.ascontiguousarray(audio, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(f"{int(sample_rate)}:{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cache_key(content_hash: str, fps: int, beats_per_bar: tuple[int, ...]) -> str:
    bars = ",".join(str(bar) for bar in beats_per_bar)
    raw = (
        f"cache={CACHE_SCHEMA_VERSION}|worker={WORKER_VERSION}|fps={int(fps)}|"
        f"bars={bars}|sha256={content_hash}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class BeatAnalysisError(ValueError):
    """The worker document is not a result this version can store."""


def parse_worker_document(document: dict, *, fps: int, beats_per_bar: tuple[int, ...]) -> dict:
    """Return a canonical document, or raise BeatAnalysisError."""
    if not isinstance(document, dict):
        raise BeatAnalysisError("worker JSON must be an object")
    _require_equal(document, "schema_version", SCHEMA_VERSION)
    _require_equal(document, "worker_version", WORKER_VERSION)
    _require_equal(document, "time_unit", TIME_UNIT)
    _require_equal(document, "beat_position_semantics", BEAT_POSITION_SEMANTICS)
    _require_equal(document, "tempo_unit", TEMPO_UNIT)
    _require_equal(document, "engine", ENGINE)
    status = document.get("status")
    if status not in {STATUS_OK, STATUS_EMPTY, STATUS_ERROR}:
        raise BeatAnalysisError(f"unexpected status: {status!r}")
    message = document.get("message", "")
    if not isinstance(message, str):
        raise BeatAnalysisError("message must be a string")
    engine_version = document.get("engine_version")
    if engine_version is not None and not isinstance(engine_version, str):
        raise BeatAnalysisError("engine_version must be a string or null")
    if document.get("fps") != int(fps):
        raise BeatAnalysisError("fps does not match the request")
    bars = document.get("beats_per_bar")
    if list(bars) != [int(bar) for bar in beats_per_bar]:
        raise BeatAnalysisError("beats_per_bar does not match the request")
    sample_rate = document.get("sample_rate")
    if sample_rate is not None and (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise BeatAnalysisError("sample_rate must be a positive integer or null")
    tempo = document.get("tempo_bpm")
    if tempo is not None:
        tempo = _finite_number(tempo, "tempo_bpm")
        if tempo <= 0.0:
            raise BeatAnalysisError("tempo_bpm must be positive")
    events = _parse_events(document.get("events"), max(beats_per_bar))
    if status == STATUS_EMPTY and events:
        raise BeatAnalysisError("empty analysis cannot contain events")
    if status == STATUS_OK and not events:
        raise BeatAnalysisError("ok analysis requires at least one event")
    if status == STATUS_ERROR:
        events = ()
        tempo = None
    if tempo is None and status == STATUS_OK and len(events) >= 2:
        raise BeatAnalysisError("tempo_bpm is required when two or more beats were detected")
    return {
        "schema_version": SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "engine": ENGINE,
        "engine_version": engine_version,
        "status": status,
        "message": message,
        "time_unit": TIME_UNIT,
        "beat_position_semantics": BEAT_POSITION_SEMANTICS,
        "tempo_unit": TEMPO_UNIT,
        "tempo_bpm": tempo,
        "fps": int(fps),
        "beats_per_bar": [int(bar) for bar in beats_per_bar],
        "sample_rate": sample_rate,
        "events": [{"time_seconds": event.time_seconds, "beat_position": event.beat_position} for event in events],
    }


def result_from_document(document: dict, *, stderr: str = "", cached: bool = False, sample_rate: int | None = None) -> BeatAnalysisResult:
    events = tuple(
        BeatEvent(float(event["time_seconds"]), int(event["beat_position"]))
        for event in document.get("events") or ()
    )
    rate = document.get("sample_rate")
    if rate is None:
        rate = sample_rate
    return BeatAnalysisResult(
        status=document["status"],
        events=events,
        tempo_bpm=document.get("tempo_bpm"),
        time_unit=document["time_unit"],
        beat_position_semantics=document["beat_position_semantics"],
        tempo_unit=document["tempo_unit"],
        schema_version=int(document["schema_version"]),
        worker_version=str(document["worker_version"]),
        engine=document["engine"],
        engine_version=document.get("engine_version"),
        fps=document.get("fps"),
        beats_per_bar=tuple(int(bar) for bar in document.get("beats_per_bar") or ()),
        sample_rate=rate,
        message=document.get("message") or "",
        stderr=stderr,
        cached=cached,
    )


class BeatAnalysisClient:
    """Run the Madmom worker, validate its JSON, and cache successes."""

    def __init__(
        self,
        python: Path | str | None = None,
        worker_script: Path | str | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 300.0,
    ):
        self.python = Path(python) if python is not None else default_madmom_python()
        self.worker_script = Path(worker_script) if worker_script is not None else default_worker_script()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.timeout = float(timeout)

    def analyze(self, source, sample_rate: int | None = None, *, path=None, **kwargs) -> BeatAnalysisResult:
        """Analyze a file path or an in-memory array.

        A path is handed to the worker as-is. A temporary WAV is written only
        for samples that have no source file, and only after a cache miss.
        """
        if path is not None:
            return self.analyze_path(path, sample_rate=sample_rate, **kwargs)
        if isinstance(source, (str, os.PathLike)):
            return self.analyze_path(source, sample_rate=sample_rate, **kwargs)
        if sample_rate is None:
            raise ValueError("sample_rate is required for in-memory audio")
        return self.analyze_samples(source, int(sample_rate), **kwargs)

    def analyze_path(self, path, *, sample_rate: int | None = None, fps: int = 100, beats_per_bar=(3, 4), timeout: float | None = None, on_progress=None, cancel_event: threading.Event | None = None) -> BeatAnalysisResult:
        audio_path = Path(path).expanduser().resolve()
        fps, bars = _request(fps, beats_per_bar)
        if not audio_path.is_file():
            return _failure(STATUS_ERROR, f"audio file not found: {audio_path}", fps, bars)
        content_hash = file_sha256(audio_path)
        cached = self._read_cache(content_hash, fps, bars, sample_rate)
        if cached is not None:
            return cached
        if _cancelled(cancel_event):
            return _failure(STATUS_CANCELLED, "analysis cancelled", fps, bars)
        return self._execute(
            audio_path,
            content_hash,
            fps,
            bars,
            sample_rate,
            timeout,
            on_progress,
            cancel_event,
        )

    def analyze_samples(self, audio, sample_rate: int, *, fps: int = 100, beats_per_bar=(3, 4), timeout: float | None = None, on_progress=None, cancel_event: threading.Event | None = None) -> BeatAnalysisResult:
        fps, bars = _request(fps, beats_per_bar)
        content_hash = samples_sha256(audio, sample_rate)
        cached = self._read_cache(content_hash, fps, bars, sample_rate)
        if cached is not None:
            return cached
        if _cancelled(cancel_event):
            return _failure(STATUS_CANCELLED, "analysis cancelled", fps, bars)
        import soundfile as sf

        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        descriptor, name = tempfile.mkstemp(prefix="soniclab-beats-", suffix=".wav")
        os.close(descriptor)
        wav_path = Path(name)
        try:
            sf.write(wav_path, array, int(sample_rate), subtype="FLOAT")
            return self._execute(
                wav_path,
                content_hash,
                fps,
                bars,
                int(sample_rate),
                timeout,
                on_progress,
                cancel_event,
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def _execute(self, audio_path, content_hash, fps, bars, sample_rate, timeout, on_progress, cancel_event) -> BeatAnalysisResult:
        python_ready = self.python is not None and Path(self.python).is_file()
        worker_ready = self.worker_script.is_file()
        if not python_ready or not worker_ready:
            looked = ", ".join(str(path) for path in _madmom_python_candidates())
            if not python_ready:
                message = f"Madmom is not installed. SonicLab looked for its Python at {looked}."
            else:
                message = f"Madmom's worker script is missing: {self.worker_script}."
            _record_madmom(message)
            return _failure(STATUS_UNAVAILABLE, message, fps, bars)
        command = [
            str(self.python),
            str(self.worker_script),
            str(audio_path),
            "--fps",
            str(fps),
            "--bars",
            *[str(bar) for bar in bars],
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return _failure(STATUS_UNAVAILABLE, str(exc), fps, bars)
        stdout, stderr, stop_status = _communicate(process, self.timeout if timeout is None else timeout, on_progress, cancel_event)
        if stop_status is not None:
            return _failure(stop_status, _stop_message(stop_status), fps, bars, stderr=stderr)
        try:
            document = parse_worker_document(_extract_json(stdout), fps=fps, beats_per_bar=bars)
        except (BeatAnalysisError, json.JSONDecodeError, ValueError) as exc:
            detail = str(exc)
            if process.returncode not in (0, None) and stderr:
                detail = f"{detail}; worker exit {process.returncode}"
            return _failure(STATUS_ERROR, detail, fps, bars, stderr=stderr)
        if process.returncode not in (0, None):
            detail = document.get("message") or ""
            exit_note = f"worker exit {process.returncode}"
            message = f"{detail} ({exit_note})" if detail else exit_note
            return _failure(STATUS_ERROR, message, fps, bars, stderr=stderr)
        if sample_rate is not None and document.get("sample_rate") is None:
            document["sample_rate"] = int(sample_rate)
        result = result_from_document(document, stderr=stderr, sample_rate=sample_rate)
        if result.status in _CACHEABLE:
            self._write_cache(content_hash, fps, bars, document)
        return result

    def _read_cache(self, content_hash, fps, bars, sample_rate) -> BeatAnalysisResult | None:
        path = self._cache_path(content_hash, fps, bars)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict):
            return None
        if envelope.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if envelope.get("worker_version") != WORKER_VERSION:
            return None
        if envelope.get("content_sha256") != content_hash:
            return None
        if envelope.get("fps") != int(fps) or list(envelope.get("beats_per_bar") or []) != list(bars):
            return None
        try:
            document = parse_worker_document(envelope.get("result"), fps=fps, beats_per_bar=bars)
        except BeatAnalysisError:
            return None
        if document["status"] not in _CACHEABLE:
            return None
        return result_from_document(document, cached=True, sample_rate=sample_rate)

    def _write_cache(self, content_hash, fps, bars, document: dict) -> None:
        path = self._cache_path(content_hash, fps, bars)
        envelope = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "worker_version": WORKER_VERSION,
            "content_sha256": content_hash,
            "fps": int(fps),
            "beats_per_bar": list(bars),
            "result": document,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(envelope), encoding="utf-8")
        os.replace(temporary, path)

    def _cache_path(self, content_hash, fps, bars) -> Path:
        return self.cache_dir / f"{cache_key(content_hash, fps, bars)}.json"


def _request(fps, beats_per_bar) -> tuple[int, tuple[int, ...]]:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    bars = tuple(int(bar) for bar in beats_per_bar)
    if not bars or any(isinstance(bar, bool) or bar < 1 for bar in beats_per_bar):
        raise ValueError("beats_per_bar must contain positive integers")
    return fps, bars


def _failure(status: str, message: str, fps: int, bars: tuple[int, ...], stderr: str = "") -> BeatAnalysisResult:
    return BeatAnalysisResult(
        status=status,
        message=message,
        fps=fps,
        beats_per_bar=bars,
        stderr=stderr,
    )


def _cancelled(cancel_event) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _stop_message(status: str) -> str:
    if status == STATUS_TIMEOUT:
        return "analysis timed out"
    if status == STATUS_CANCELLED:
        return "analysis cancelled"
    return status


def _parse_events(raw, maximum_position: int) -> tuple[BeatEvent, ...]:
    if not isinstance(raw, list):
        raise BeatAnalysisError("events must be a list")
    events = []
    previous = None
    for item in raw:
        if not isinstance(item, dict):
            raise BeatAnalysisError("each event must be an object")
        seconds = _finite_number(item.get("time_seconds"), "time_seconds")
        if seconds < 0.0:
            raise BeatAnalysisError("time_seconds must be >= 0")
        if previous is not None and seconds <= previous:
            raise BeatAnalysisError("time_seconds must be strictly increasing")
        position = item.get("beat_position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise BeatAnalysisError("beat_position must be an integer >= 1")
        if position > maximum_position:
            raise BeatAnalysisError("beat_position is outside beats_per_bar")
        events.append(BeatEvent(seconds, position))
        previous = seconds
    return tuple(events)


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BeatAnalysisError(f"{label} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise BeatAnalysisError(f"{label} must be finite")
    return number


def _require_equal(document: dict, key: str, expected) -> None:
    if document.get(key) != expected:
        raise BeatAnalysisError(f"{key} must be {expected!r}")


def _extract_json(text: str) -> dict:
    body = text.strip()
    if not body:
        raise BeatAnalysisError("worker produced no JSON")
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        start = body.find("{")
        end = body.rfind("}")
        if start < 0 or end <= start:
            raise BeatAnalysisError("worker JSON is invalid") from None
        try:
            value = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise BeatAnalysisError("worker JSON is invalid") from exc
    if not isinstance(value, dict):
        raise BeatAnalysisError("worker JSON must be an object")
    return value


def _communicate(process: subprocess.Popen, timeout: float, on_progress, cancel_event) -> tuple[str, str, str | None]:
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    stdout_thread = threading.Thread(target=_drain, args=(process.stdout, stdout_parts, _STDOUT_LIMIT, None), daemon=True)
    stderr_thread = threading.Thread(
        target=_drain,
        args=(process.stderr, stderr_parts, _STDERR_LIMIT, on_progress),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + max(0.0, float(timeout))
    stop_status = None
    while process.poll() is None:
        if _cancelled(cancel_event):
            stop_status = STATUS_CANCELLED
            break
        if time.monotonic() >= deadline:
            stop_status = STATUS_TIMEOUT
            break
        time.sleep(0.02)
    if stop_status is not None:
        _kill(process)
    else:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill(process)
            stop_status = STATUS_TIMEOUT
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    stdout = b"".join(stdout_parts).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_parts).decode("utf-8", errors="replace")
    return stdout, stderr, stop_status


def _drain(pipe, parts: list[bytes], limit: int, on_progress) -> None:
    if pipe is None:
        return
    kept = 0
    pending = b""
    try:
        while True:
            chunk = pipe.readline()
            if not chunk:
                break
            if on_progress is not None:
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    _report_stage(line.decode("utf-8", errors="replace"), on_progress)
            if kept < limit:
                parts.append(chunk[: limit - kept])
                kept += len(chunk)
    finally:
        pipe.close()


def _report_stage(line: str, on_progress) -> None:
    text = line.strip()
    if not text.startswith("stage "):
        return
    stage = text[6:].strip()
    if not stage:
        return
    try:
        on_progress(stage)
    except Exception:
        return


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
