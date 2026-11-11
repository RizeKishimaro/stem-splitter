"""Split songs into stems with Demucs in a separate process.

Demucs (PyTorch) lives in its own environment, like the Madmom worker. This
module only starts ``python -m demucs`` and reads its progress output.
It never imports torch.

Output
    ``<folder>/<song>-<sha12>/<stem>.wav`` for drums, bass, other and vocals.
    A folder that already holds every stem is reused without running Demucs.

Progress
    Demucs prints ``Selected model is a bag of N models`` and then one tqdm
    bar per model. The overall fraction is ``(finished bars + current) / N``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from soniclab.core.beat_analysis import file_sha256

MODEL = "htdemucs_ft"
STEMS = ("drums", "bass", "other", "vocals")
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_UNAVAILABLE = "unavailable"
STATUS_CANCELLED = "cancelled"

# Names that already are a stem. These are not split again.
STEM_WORDS = {"drums", "drum", "bass", "other", "vocals", "vocal", "voice", "piano", "guitar", "accompaniment", "instrumental"}

_BAG = re.compile(r"bag of (\d+) models?")
_PERCENT = re.compile(r"(\d{1,3})%\|")
_STDERR_LIMIT = 16_384


@dataclass(frozen=True)
class StemResult:
    status: str
    stems: dict = field(default_factory=dict)
    message: str = ""
    cached: bool = False


def default_demucs_python() -> Path | None:
    """Interpreter that can run ``python -m demucs``, if this machine has one."""
    override = os.environ.get("SONICLAB_DEMUCS_PYTHON")
    if override is not None:
        # Set but empty turns Demucs off, which the tests rely on.
        if not override:
            return None
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in (
        Path(__file__).resolve().parents[2] / ".venv-demucs" / "bin" / "python",
        Path.home() / ".venvs" / "demucs" / "bin" / "python",
    ):
        if candidate.is_file():
            return candidate
    return None


def default_stem_folder(project_path: str | None = None) -> Path:
    """``Stems`` beside a saved project, else a shared user folder."""
    if project_path:
        return Path(project_path).expanduser().resolve().parent / "Stems"
    return Path.home() / ".local" / "share" / "soniclab" / "stems"


def is_stem_name(name: str) -> bool:
    """True for ``drums``, ``Song vocals`` or ``bass_2``, not for ``Piano Man``."""
    words = [word for word in re.split(r"[^a-z]+", str(name or "").lower()) if word]
    return bool(words) and words[-1] in STEM_WORDS


def parse_progress(line: str, state: dict) -> float | None:
    """Update ``state`` from one output line. Returns the overall fraction."""
    bag = _BAG.search(line)
    if bag:
        state["models"] = max(1, int(bag.group(1)))
        return None
    match = _PERCENT.search(line)
    if not match:
        return None
    percent = min(100, int(match.group(1)))
    last = state.get("percent", 0)
    if percent < last:
        state["done"] = state.get("done", 0) + 1
    state["percent"] = percent
    models = max(1, int(state.get("models", 1)))
    done = min(state.get("done", 0), models - 1)
    return min(1.0, (done + percent / 100.0) / models)


def stem_paths(folder: Path) -> dict:
    return {stem: folder / f"{stem}.wav" for stem in STEMS}


def split_folder(path, folder) -> Path:
    """Where SonicLab keeps the stems of ``path``: ``<folder>/<song>-<sha12>``."""
    source = Path(path).expanduser().resolve()
    return Path(folder) / f"{source.stem}-{file_sha256(source)[:12]}"


def existing_stems(path, folder, model: str = MODEL) -> dict | None:
    """Stems made earlier for this song, or None.

    Checks SonicLab's own folder first, then the layout plain ``demucs``
    writes when run beside the song: ``<song dir>/separated/<model>/<song>/``.
    """
    source = Path(path).expanduser().resolve()
    try:
        candidates = [split_folder(source, folder)]
    except OSError:
        return None
    for name in dict.fromkeys((model, MODEL, "htdemucs")):
        candidates.append(source.parent / "separated" / name / source.stem)
    for candidate in candidates:
        wanted = stem_paths(candidate)
        if all(item.is_file() for item in wanted.values()):
            return {stem: str(item) for stem, item in wanted.items()}
    return None


class StemSplitter:
    """Run Demucs on one file. Safe to call from a worker thread."""

    def __init__(self, python: Path | str | None = None, model: str = MODEL):
        self.python = Path(python) if python is not None else default_demucs_python()
        self.model = model

    @property
    def available(self) -> bool:
        return self.python is not None and Path(self.python).is_file()

    def split(self, path, folder, on_progress=None, cancel_event: threading.Event | None = None) -> StemResult:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            return StemResult(STATUS_ERROR, message=f"audio file not found: {source}")
        try:
            target = split_folder(source, folder)
        except OSError as exc:
            return StemResult(STATUS_ERROR, message=str(exc))
        found = existing_stems(source, folder, self.model)
        if found is not None:
            return StemResult(STATUS_OK, found, cached=True)
        wanted = stem_paths(target)
        if not self.available:
            return StemResult(STATUS_UNAVAILABLE, message="Demucs is not installed")
        target.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            "-m",
            "demucs",
            "-n",
            self.model,
            "-o",
            str(target),
            "--filename",
            "{stem}.{ext}",
            str(source),
        ]
        try:
            process = subprocess.Popen(
                command,
                # The "bag of N models" line goes to stdout, the bars to stderr.
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return StemResult(STATUS_UNAVAILABLE, message=str(exc))
        tail = self._follow(process, on_progress, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return StemResult(STATUS_CANCELLED, message="splitting cancelled")
        if process.returncode != 0:
            detail = tail.strip().splitlines()[-1] if tail.strip() else ""
            return StemResult(STATUS_ERROR, message=f"Demucs exit {process.returncode}: {detail}".strip())
        # Demucs writes into <out>/<model>/. Move the stems up one level.
        produced = target / self.model
        for name, final in wanted.items():
            made = produced / f"{name}.wav"
            if made.is_file():
                os.replace(made, final)
        try:
            produced.rmdir()
        except OSError:
            pass
        missing = [name for name, item in wanted.items() if not item.is_file()]
        if missing:
            return StemResult(STATUS_ERROR, message=f"Demucs did not write {', '.join(missing)}")
        return StemResult(STATUS_OK, {name: str(item) for name, item in wanted.items()})

    def _follow(self, process, on_progress, cancel_event) -> str:
        state: dict = {}
        tail = ""
        stop = threading.Event()

        def watch():
            while not stop.wait(0.1):
                if cancel_event is not None and cancel_event.is_set():
                    _kill(process)
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        pending = b""
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                pending += chunk
                # tqdm redraws with carriage returns, not new lines.
                parts = re.split(rb"[\r\n]", pending)
                pending = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", errors="replace")
                    if line.strip():
                        tail = (tail + line + "\n")[-_STDERR_LIMIT:]
                    fraction = parse_progress(line, state)
                    if fraction is not None and on_progress is not None:
                        try:
                            on_progress(fraction)
                        except Exception:
                            pass
        finally:
            process.stdout.close()
            process.wait()
            stop.set()
            watcher.join(timeout=1)
        return tail


class StemJobQueue:
    """One Demucs run at a time. Messages are polled from the UI thread.

    ``("progress", job_id, fraction)`` and ``("finished", job_id, StemResult)``.
    """

    def __init__(self, splitter: StemSplitter | None = None):
        self._splitter = splitter if splitter is not None else StemSplitter()
        self._lock = threading.Lock()
        self._jobs: list[tuple[str, str, str]] = []
        self._messages: list[tuple] = []
        self._cancel: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._active: str | None = None
        self._counter = 0

    @property
    def available(self) -> bool:
        return self._splitter.available

    def submit(self, path, folder) -> str:
        with self._lock:
            self._counter += 1
            job_id = f"stems-{self._counter}"
            self._jobs.append((job_id, str(path), str(folder)))
        self._ensure_worker()
        return job_id

    def pending(self) -> int:
        with self._lock:
            return len(self._jobs) + (1 if self._active else 0)

    def poll(self):
        with self._lock:
            if not self._messages:
                return None
            return self._messages.pop(0)

    def cancel(self) -> None:
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
            dropped = [job_id for job_id, _path, _folder in self._jobs]
            self._jobs.clear()
            for job_id in dropped:
                self._messages.append(("finished", job_id, StemResult(STATUS_CANCELLED, message="splitting cancelled")))

    def drain(self, timeout: float = 5.0) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._work, name="soniclab-stems", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self._jobs:
                    self._active = None
                    self._worker = None
                    return
                job_id, path, folder = self._jobs.pop(0)
                self._active = job_id
                cancel = self._cancel = threading.Event()
            try:
                result = self._splitter.split(
                    path,
                    folder,
                    on_progress=lambda fraction, job=job_id: self._publish(("progress", job, fraction)),
                    cancel_event=cancel,
                )
            except Exception as exc:
                result = StemResult(STATUS_ERROR, message=str(exc))
            with self._lock:
                self._active = None
                self._cancel = None
            self._publish(("finished", job_id, result))

    def _publish(self, message: tuple) -> None:
        with self._lock:
            self._messages.append(message)


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
