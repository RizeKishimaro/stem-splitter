"""Split songs into stems with Demucs in a separate process.

Demucs (PyTorch) lives in its own environment, like the Madmom worker. This
module only starts ``python -m demucs`` and reads its progress output.
It never imports torch.

Models
    ``htdemucs_ft`` (drums, bass, other, vocals) is the default and what
    "split on import" uses. ``htdemucs_6s`` adds guitar and piano. Demucs has
    no snare stem: the snare is part of drums.

Output
    ``<folder>/<song>-<sha12>/<stem>.wav`` for the default model and
    ``<folder>/<song>-<sha12>/<model>/<stem>.wav`` for the others.
    A folder that already holds every stem is reused without running Demucs.
    Demucs writes into a ``.partial`` folder first, so a cancelled or failed
    run never leaves stems that look finished.

Devices
    ``auto`` lets Demucs choose (an NVIDIA GPU when it finds one, else the
    CPU). ``mps`` is Apple's GPU backend. It is not verified with Demucs, so
    it is never chosen automatically, and a failed ``mps`` run is repeated
    on the CPU.

Progress
    Demucs prints ``Selected model is a bag of N models`` and then one tqdm
    bar per model. The overall fraction is ``(finished bars + current) / N``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from soniclab.core.audio_load import AudioLoadError, explain_separation_error, prepare_demucs_input
from soniclab.core.beat_analysis import file_sha256



@dataclass(frozen=True)
class StemModel:
    name: str
    label: str
    stems: tuple[str, ...]
    passes: int  # Demucs models run per song; each one draws a progress bar
    note: str = ""


MODEL = "htdemucs_ft"
SIX_STEM_MODEL = "htdemucs_6s"
MODELS = {
    SIX_STEM_MODEL: StemModel(
        SIX_STEM_MODEL,
        "6 stems (htdemucs_6s)",
        ("vocals", "drums", "bass", "guitar", "piano", "other"),
        1,
        "Piano is its own stem. The snare stays inside drums; nothing here is labeled snare. "
        "Demucs calls this model experimental, and piano is its weakest stem.",
    ),
    MODEL: StemModel(
        MODEL,
        "4 stems, best quality (htdemucs_ft)",
        ("drums", "bass", "other", "vocals"),
        4,
        "Four passes per song, so it takes about four times as long.",
    ),
    "htdemucs": StemModel("htdemucs", "4 stems, fast (htdemucs)", ("drums", "bass", "other", "vocals"), 1),
}
STEMS = MODELS[MODEL].stems

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
DEVICE_LABELS = {
    DEVICE_AUTO: "Auto",
    DEVICE_CPU: "CPU",
    DEVICE_CUDA: "NVIDIA GPU (CUDA)",
    DEVICE_MPS: "Apple GPU (MPS, experimental)",
}
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
    model: str = ""
    device: str = ""  # what Demucs was asked to use; empty for cached stems


@dataclass(frozen=True)
class EngineInfo:
    """What ``probe_engine`` found in the Demucs environment."""

    available: bool
    message: str = ""
    demucs_version: str = ""
    torch_version: str = ""
    devices: tuple[str, ...] = (DEVICE_AUTO, DEVICE_CPU)
    # Model name -> True when its weights are already downloaded, False when
    # Demucs will download them on first use, None when unknown.
    downloaded: dict = field(default_factory=dict)


def model_stems(model: str = MODEL) -> tuple[str, ...]:
    found = MODELS.get(model)
    return found.stems if found is not None else STEMS


def track_order(stems) -> tuple[str, ...]:
    """Order for new tracks: drums first, because they lead the downbeats."""
    names = tuple(stems)
    return tuple(name for name in names if name == "drums") + tuple(name for name in names if name != "drums")


def default_demucs_python() -> Path | None:
    """Interpreter that can run ``python -m demucs``, if this machine has one."""
    override = os.environ.get("SONICLAB_DEMUCS_PYTHON")
    if override is not None:
        # Set but empty turns Demucs off, which the tests rely on.
        if not override:
            return None
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    installed = Path.home() / "Library" / "Application Support" / "SonicLab" / "demucs-venv"
    candidates = []
    if (installed / ".soniclab-ready").is_file():
        candidates.append(installed / "bin" / "python")
    candidates.extend((
        Path(__file__).resolve().parents[2] / ".venv-demucs" / "bin" / "python",
        Path.home() / ".venvs" / "demucs" / "bin" / "python",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_stem_folder(project_path: str | None = None) -> Path:
    """``Stems`` beside a saved project, else a shared user folder."""
    if project_path:
        return Path(project_path).expanduser().resolve().parent / "Stems"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SonicLab" / "stems"
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


def stem_paths(folder: Path, model: str = MODEL) -> dict:
    return {stem: Path(folder) / f"{stem}.wav" for stem in model_stems(model)}


def split_folder(path, folder, model: str = MODEL) -> Path:
    """Where SonicLab keeps the stems of ``path``.

    ``<folder>/<song>-<sha12>`` for the default model, with a ``<model>``
    folder inside it for any other, so stem sets never mix.
    """
    source = Path(path).expanduser().resolve()
    base = Path(folder) / f"{source.stem}-{file_sha256(source)[:12]}"
    return base if model == MODEL else base / model


def existing_stems(path, folder, model: str = MODEL) -> dict | None:
    """Stems made earlier for this song, or None.

    Checks SonicLab's own folder first, then the layout plain ``demucs``
    writes when run beside the song: ``<song dir>/separated/<model>/<song>/``.
    """
    source = Path(path).expanduser().resolve()
    try:
        candidates = [split_folder(source, folder, model)]
    except OSError:
        return None
    # Any model with the same stems will do for a song split by hand.
    same = [name for name, known in MODELS.items() if set(known.stems) == set(model_stems(model))]
    for name in dict.fromkeys((model, *same)):
        candidates.append(source.parent / "separated" / name / source.stem)
    for candidate in candidates:
        wanted = stem_paths(candidate, model)
        if all(item.is_file() for item in wanted.values()):
            return {stem: str(item) for stem, item in wanted.items()}
    return None


def export_stems(stems: dict, folder, song: str) -> list[Path]:
    """Copy stems to ``<folder>/<song> - <stem>.wav``. Never overwrites."""
    target = Path(folder).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, source in stems.items():
        destination = target / f"{song} - {stem}.wav"
        number = 2
        while destination.exists():
            destination = target / f"{song} - {stem} ({number}).wav"
            number += 1
        shutil.copyfile(source, destination)
        written.append(destination)
    return written


_PROBE = """
import json, sys
out = {}
try:
    import torch, demucs
    out["torch"] = str(torch.__version__)
    out["demucs"] = str(demucs.__version__)
    out["cuda"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    out["mps"] = bool(mps is not None and mps.is_available())
    weights = {}
    try:
        import os, pathlib, yaml
        from demucs import pretrained
        hub = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
        remote = pathlib.Path(pretrained.__file__).parent / "remote"
        caches = []
        for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
            if os.environ.get(key):
                caches.append(pathlib.Path(os.environ[key]))
        if os.environ.get("HF_HOME"):
            caches.append(pathlib.Path(os.environ["HF_HOME"]) / "hub")
        caches.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")

        def present(sig):
            if any(hub.glob(sig + "-*.th")) or any(hub.glob(sig + ".safetensors")):
                return True
            seen = set()
            for cache in caches:
                folder = str(cache)
                if folder in seen or not cache.is_dir():
                    continue
                seen.add(folder)
                if any(cache.glob("**/" + sig + ".safetensors")):
                    return True
            return False

        for name in sys.argv[1:]:
            try:
                signatures = yaml.safe_load((remote / (name + ".yaml")).read_text())["models"]
                weights[name] = all(present(sig) for sig in signatures)
            except Exception:
                weights[name] = None
    except Exception:
        pass
    out["weights"] = weights
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print("SONICLAB_PROBE " + json.dumps(out))
"""


def probe_engine(python: Path | str | None = None, timeout: float = 120.0) -> EngineInfo:
    """Ask the Demucs environment what it has. Slow (imports PyTorch): call
    it from a worker thread, never from the UI thread."""
    interpreter = Path(python) if python is not None else default_demucs_python()
    if interpreter is None or not Path(interpreter).is_file():
        return EngineInfo(False, "Demucs is not installed")
    try:
        done = subprocess.run(
            [str(interpreter), "-c", _PROBE, *MODELS],
            capture_output=True,
            timeout=timeout,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EngineInfo(False, f"Could not start the Demucs environment: {exc}")
    document = None
    for line in done.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("SONICLAB_PROBE "):
            try:
                document = json.loads(line[len("SONICLAB_PROBE "):])
            except ValueError:
                document = None
    if not isinstance(document, dict):
        return EngineInfo(False, "The Demucs environment did not answer")
    if document.get("error"):
        return EngineInfo(False, f"Demucs is not installed in {interpreter.parent.parent}: {document['error']}")
    devices = [DEVICE_AUTO, DEVICE_CPU]
    if document.get("cuda"):
        devices.append(DEVICE_CUDA)
    if document.get("mps"):
        devices.append(DEVICE_MPS)
    weights = document.get("weights") if isinstance(document.get("weights"), dict) else {}
    return EngineInfo(
        True,
        "",
        str(document.get("demucs", "")),
        str(document.get("torch", "")),
        tuple(devices),
        {name: weights.get(name) for name in MODELS},
    )


class StemSplitter:
    """Run Demucs on one file. Safe to call from a worker thread."""

    def __init__(self, python: Path | str | None = None, model: str = MODEL):
        self.python = Path(python) if python is not None else default_demucs_python()
        self.model = model

    @property
    def available(self) -> bool:
        return self.python is not None and Path(self.python).is_file()

    def split(
        self,
        path,
        folder,
        on_progress=None,
        cancel_event: threading.Event | None = None,
        *,
        model: str | None = None,
        device: str | None = None,
        segment: int | None = None,
    ) -> StemResult:
        """Split ``path`` with ``model`` (default: this splitter's model).

        ``device`` is ``auto``, ``cpu``, ``cuda`` or ``mps``. ``segment`` is
        Demucs's ``--segment`` in seconds; smaller uses less memory.
        """
        model = model or self.model
        device = device or DEVICE_AUTO
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            return StemResult(STATUS_ERROR, message=f"Audio file not found: {source}", model=model)
        try:
            target = split_folder(source, folder, model)
        except OSError as exc:
            return StemResult(STATUS_ERROR, message=str(exc), model=model)
        found = existing_stems(source, folder, model)
        if found is not None:
            return StemResult(STATUS_OK, found, cached=True, model=model)
        wanted = stem_paths(target, model)
        if not self.available:
            return StemResult(STATUS_UNAVAILABLE, message="Demucs is not installed", model=model)
        target.mkdir(parents=True, exist_ok=True)
        try:
            prepared = prepare_demucs_input(source, target)
        except AudioLoadError as exc:
            return StemResult(STATUS_ERROR, message=str(exc), model=model)
        # Demucs writes into <partial>/<model>/. Stems move out only when
        # the whole run succeeded.
        partial = target / ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        attempts = [device, DEVICE_CPU] if device == DEVICE_MPS else [device]
        note = ""
        used = device
        try:
            for attempt in attempts:
                used = attempt
                code, tail = self._run(prepared.path, partial, model, attempt, segment, on_progress, cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    return StemResult(STATUS_CANCELLED, message="splitting cancelled", model=model, device=used)
                if code == 0:
                    break
                if attempt != attempts[-1]:
                    note = "The Apple GPU run failed, so the CPU was used"
                    shutil.rmtree(partial, ignore_errors=True)
            if code is None:
                return StemResult(STATUS_UNAVAILABLE, message=tail, model=model, device=used)
            if code != 0:
                detail = tail.strip().splitlines()[-1] if tail.strip() else ""
                decoded = explain_separation_error(tail) or explain_separation_error(detail)
                if decoded:
                    return StemResult(STATUS_ERROR, message=decoded, model=model, device=used)
                return StemResult(
                    STATUS_ERROR, message=f"Demucs exit {code}: {detail}".strip(), model=model, device=used
                )
            produced = partial / model
            for name, final in wanted.items():
                made = produced / f"{name}.wav"
                if made.is_file():
                    os.replace(made, final)
        finally:
            prepared.cleanup()
            shutil.rmtree(partial, ignore_errors=True)
        missing = [name for name, item in wanted.items() if not item.is_file()]
        if missing:
            return StemResult(
                STATUS_ERROR, message=f"Demucs did not write {', '.join(missing)}", model=model, device=used
            )
        return StemResult(
            STATUS_OK, {name: str(item) for name, item in wanted.items()}, message=note, model=model, device=used
        )

    def _run(self, source, out, model, device, segment, on_progress, cancel_event) -> tuple[int | None, str]:
        """One Demucs process. Returns its exit code (None if it never
        started) and the end of its output."""
        command = [str(self.python), "-m", "demucs", "-n", model, "-o", str(out), "--filename", "{stem}.{ext}"]
        if device and device != DEVICE_AUTO:
            command += ["-d", device]
        if segment:
            command += ["--segment", str(int(segment))]
        command.append(str(source))
        environment = dict(os.environ)
        if device == DEVICE_MPS:
            # Operations PyTorch lacks on MPS run on the CPU instead of failing.
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        try:
            process = subprocess.Popen(
                command,
                # The "bag of N models" line goes to stdout, the bars to stderr.
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        except OSError as exc:
            return None, str(exc)
        tail = self._follow(process, on_progress, cancel_event)
        return process.returncode, tail

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
            # Output that ended without a line break still counts.
            last = pending.decode("utf-8", errors="replace")
            if last.strip():
                tail = (tail + last + "\n")[-_STDERR_LIMIT:]
                fraction = parse_progress(last, state)
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
        self._jobs: list[tuple[str, str, str, dict]] = []
        self._messages: list[tuple] = []
        self._cancel: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._active: str | None = None
        self._counter = 0

    @property
    def available(self) -> bool:
        return self._splitter.available

    @property
    def python(self) -> Path | None:
        return self._splitter.python

    def submit(self, path, folder, *, model: str | None = None, device: str | None = None, segment: int | None = None) -> str:
        options = {"model": model, "device": device, "segment": segment}
        with self._lock:
            self._counter += 1
            job_id = f"stems-{self._counter}"
            self._jobs.append((job_id, str(path), str(folder), options))
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
            dropped = [job[0] for job in self._jobs]
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
                job_id, path, folder, options = self._jobs.pop(0)
                self._active = job_id
                cancel = self._cancel = threading.Event()
            try:
                result = self._splitter.split(
                    path,
                    folder,
                    on_progress=lambda fraction, job=job_id: self._publish(("progress", job, fraction)),
                    cancel_event=cancel,
                    **options,
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
