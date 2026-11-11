"""One-at-a-time Madmom analysis for source files.

The worker thread is the only caller of BeatAnalysisClient. Results stay in
this process, keyed by the same file SHA-256 the client uses for its cache.
"""

from __future__ import annotations

import threading
from pathlib import Path

from soniclab.core.beat_analysis import (
    STATUS_CANCELLED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    BeatAnalysisClient,
    BeatAnalysisResult,
    file_sha256,
)


def resolve_source(path) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return None


class BeatJobQueue:
    def __init__(self, client: BeatAnalysisClient | None = None):
        self._client = client if client is not None else BeatAnalysisClient()
        self._lock = threading.Lock()
        self._paths: list[str] = []
        self._queued: set[str] = set()
        self._inflight: str | None = None
        self._known: dict[str, BeatAnalysisResult] = {}
        self._path_hash: dict[str, str] = {}
        self._messages: list[tuple] = []
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None

    def submit(self, path) -> str:
        """Queue a source file. Returns queued, duplicate, ready, or missing."""
        resolved = resolve_source(path)
        if resolved is None or not Path(resolved).is_file():
            return "missing"
        with self._lock:
            digest = self._path_hash.get(resolved)
            known = self._known.get(digest) if digest else None
            if known is not None and known.status in (STATUS_OK, STATUS_EMPTY):
                return "ready"
            if known is not None:
                self._known.pop(digest, None)
            if resolved == self._inflight or resolved in self._queued:
                return "duplicate"
            self._paths.append(resolved)
            self._queued.add(resolved)
        self._ensure_worker()
        return "queued"

    def poll(self):
        with self._lock:
            if not self._messages:
                return None
            return self._messages.pop(0)

    def events_for(self, path) -> tuple[tuple[float, int], ...]:
        """Source beats for a finished successful analysis, otherwise nothing."""
        resolved = resolve_source(path)
        if resolved is None:
            return ()
        with self._lock:
            digest = self._path_hash.get(resolved)
            result = self._known.get(digest) if digest else None
        if result is None or result.status != STATUS_OK:
            return ()
        return tuple((event.time_seconds, event.beat_position) for event in result.events)

    def result_for(self, path) -> BeatAnalysisResult | None:
        resolved = resolve_source(path)
        if resolved is None:
            return None
        with self._lock:
            digest = self._path_hash.get(resolved)
            if digest is None:
                return None
            return self._known.get(digest)

    def cancel(self) -> None:
        with self._lock:
            self._cancel.set()
            self._paths.clear()
            self._queued.clear()

    def drain(self, timeout: float = 5.0) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._cancel = threading.Event()
            self._worker = threading.Thread(target=self._work, name="soniclab-beats", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            path = self._next_path()
            if path is None:
                return
            self._analyze(path)

    def _next_path(self) -> str | None:
        with self._lock:
            if self._cancel.is_set() or not self._paths:
                self._inflight = None
                return None
            path = self._paths.pop(0)
            self._queued.discard(path)
            self._inflight = path
            return path

    def _analyze(self, path: str) -> None:
        if self._cancel.is_set():
            self._finish(path, None, BeatAnalysisResult(status=STATUS_CANCELLED, message="analysis cancelled"))
            return
        try:
            digest = file_sha256(Path(path))
        except OSError as exc:
            self._finish(path, None, BeatAnalysisResult(status=STATUS_ERROR, message=str(exc)))
            return
        with self._lock:
            known = self._known.get(digest)
        if known is not None:
            self._finish(path, digest, known)
            return
        try:
            result = self._client.analyze_path(
                path,
                fps=100,
                beats_per_bar=(3, 4),
                on_progress=lambda stage, source=path: self._publish(("progress", source, stage)),
                cancel_event=self._cancel,
            )
        except Exception as exc:
            result = BeatAnalysisResult(status=STATUS_ERROR, message=str(exc))
        self._finish(path, digest, result)

    def _finish(self, path: str, digest: str | None, result: BeatAnalysisResult) -> None:
        with self._lock:
            if digest is not None:
                self._path_hash[path] = digest
                self._known[digest] = result
            if self._inflight == path:
                self._inflight = None
        self._publish(("finished", path, result))

    def _publish(self, message: tuple) -> None:
        with self._lock:
            self._messages.append(message)
