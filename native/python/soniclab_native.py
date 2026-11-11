"""ctypes binding for the Phase 3 native engine prototype (libsoniclab_engine).

The prototype lives outside the ``soniclab`` package on purpose. The Python
AudioEngine stays the default engine, and nothing in the app imports this.

Ownership, matching native/engine_capi/include/soniclab/engine_capi.h:

* ``NativeEngine`` owns one native handle and frees it once, in ``close()``.
  It also works as a context manager, and ``__del__`` is a safety net.
* ``process(buffer)`` lends a numpy array to native code for the duration
  of the call only. The array must be C-contiguous, writeable float32 of
  shape (frames, channels). It is changed in place, and native code keeps
  no pointer to it afterwards.
* ``read_output()`` and ``stats()`` return new Python-owned copies.
  Native memory is never exposed to Python.

ctypes releases the GIL for every foreign call, and the C ABI has no
callbacks. So no Python code, and no GIL acquisition, ever happens on the
native audio thread.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ABI_VERSION = 1

SL_OK = 0
_ERRORS = {
    -1: "null handle or buffer",
    -2: "argument out of range",
    -3: "audio side is busy (device thread running)",
    -4: "command queue full",
    -5: "invalid state",
}

_NATIVE_DIR = Path(__file__).resolve().parent.parent
_LIBRARY_NAMES = {"darwin": "libsoniclab_engine.dylib", "win32": "soniclab_engine.dll"}


class NativeEngineError(RuntimeError):
    def __init__(self, code: int, call: str):
        super().__init__(f"{call} failed: {_ERRORS.get(code, 'error')} ({code})")
        self.code = code


class Stats(ctypes.Structure):
    _fields_ = [
        ("blocks_processed", ctypes.c_int64),
        ("frames_processed", ctypes.c_int64),
        ("commands_applied", ctypes.c_int64),
        ("output_frames_dropped", ctypes.c_int64),
        ("max_block_nanoseconds", ctypes.c_int64),
    ]

    def as_dict(self) -> dict:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


def find_library() -> Path | None:
    """``SONICLAB_NATIVE_LIB`` if set, otherwise the first CMake preset build found."""
    explicit = os.environ.get("SONICLAB_NATIVE_LIB")
    if explicit:
        return Path(explicit)
    name = _LIBRARY_NAMES.get(sys.platform, "libsoniclab_engine.so")
    for preset in ("linux-release", "linux-debug", "macos-ninja", "linux-asan", "linux-tsan"):
        candidate = _NATIVE_DIR / "build" / preset / "engine_capi" / name
        if candidate.exists():
            return candidate
    return None


_lib = None


def load_library(path: Path | str | None = None) -> ctypes.CDLL:
    """Load and type the library once. Raises OSError when it is missing."""
    global _lib
    if _lib is not None:
        return _lib
    path = Path(path) if path else find_library()
    if path is None:
        raise OSError("libsoniclab_engine not built; see native/README.md or set SONICLAB_NATIVE_LIB")
    # CDLL, not PyDLL: the GIL is released while native code runs.
    lib = ctypes.CDLL(str(path))
    handle = ctypes.c_void_p
    f32p = ctypes.POINTER(ctypes.c_float)
    signatures = {
        "sl_engine_abi_version": (ctypes.c_uint32, []),
        "sl_engine_create": (handle, [ctypes.c_double, ctypes.c_int32, ctypes.c_int32]),
        "sl_engine_destroy": (None, [handle]),
        "sl_engine_channels": (ctypes.c_int32, [handle]),
        "sl_engine_max_block_frames": (ctypes.c_int32, [handle]),
        "sl_engine_set_output_gain": (ctypes.c_int32, [handle, ctypes.c_float]),
        "sl_engine_set_master_muted": (ctypes.c_int32, [handle, ctypes.c_int32]),
        "sl_engine_process": (ctypes.c_int32, [handle, f32p, ctypes.c_int32]),
        "sl_engine_start_device": (ctypes.c_int32, [handle, ctypes.c_float]),
        "sl_engine_stop_device": (ctypes.c_int32, [handle]),
        "sl_engine_device_running": (ctypes.c_int32, [handle]),
        "sl_engine_read_output": (ctypes.c_int32, [handle, f32p, ctypes.c_int32]),
        "sl_engine_take_peak": (ctypes.c_float, [handle]),
        "sl_engine_get_stats": (ctypes.c_int32, [handle, ctypes.POINTER(Stats)]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    version = lib.sl_engine_abi_version()
    if version != ABI_VERSION:
        raise OSError(f"{path}: ABI version {version}, expected {ABI_VERSION}")
    _lib = lib
    return lib


def _check(code: int, call: str) -> int:
    if code < 0:
        raise NativeEngineError(code, call)
    return code


class NativeEngine:
    """Master-stage prototype: output gain, master mute, clip, peak."""

    def __init__(self, sample_rate: float = 44100.0, channels: int = 2, max_block_frames: int = 512):
        self._lib = load_library()
        self._handle = self._lib.sl_engine_create(float(sample_rate), int(channels), int(max_block_frames))
        if not self._handle:
            raise ValueError("sl_engine_create rejected the arguments")
        self.sample_rate = float(sample_rate)
        self.channels = int(channels)
        self.max_block_frames = int(max_block_frames)

    # Lifetime ---------------------------------------------------------------

    def close(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            self._lib.sl_engine_destroy(handle)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):
        self.close()

    def _live(self):
        if not self._handle:
            raise ValueError("engine is closed")
        return self._handle

    # Control (queued; applied at the next block) ----------------------------

    def set_output_gain(self, gain: float) -> None:
        _check(self._lib.sl_engine_set_output_gain(self._live(), float(gain)), "set_output_gain")

    def set_master_muted(self, muted: bool) -> None:
        _check(self._lib.sl_engine_set_master_muted(self._live(), 1 if muted else 0), "set_master_muted")

    # Offline processing (borrowed buffer) -----------------------------------

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """Process ``buffer`` in place and return it. See the module docstring."""
        if not isinstance(buffer, np.ndarray) or buffer.dtype != np.float32:
            raise TypeError("buffer must be a float32 numpy array")
        if buffer.ndim != 2 or buffer.shape[1] != self.channels:
            raise ValueError(f"buffer must have shape (frames, {self.channels})")
        if not (buffer.flags.c_contiguous and buffer.flags.writeable):
            raise ValueError("buffer must be C-contiguous and writeable")
        pointer = buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _check(self._lib.sl_engine_process(self._live(), pointer, int(buffer.shape[0])), "process")
        return buffer

    # Stand-in device thread --------------------------------------------------

    def start_device(self, tone_hz: float = 440.0) -> None:
        _check(self._lib.sl_engine_start_device(self._live(), float(tone_hz)), "start_device")

    def stop_device(self) -> None:
        _check(self._lib.sl_engine_stop_device(self._live()), "stop_device")

    @property
    def device_running(self) -> bool:
        return _check(self._lib.sl_engine_device_running(self._live()), "device_running") == 1

    def read_output(self, max_frames: int = 4096) -> np.ndarray:
        """Queued device output, oldest first, as a new (frames, channels) array."""
        out = np.empty((int(max_frames), self.channels), dtype=np.float32)
        pointer = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        count = _check(self._lib.sl_engine_read_output(self._live(), pointer, int(max_frames)), "read_output")
        return out[:count].copy()

    # Meters ------------------------------------------------------------------

    def take_peak(self) -> float:
        return float(self._lib.sl_engine_take_peak(self._live()))

    def stats(self) -> dict:
        stats = Stats()
        _check(self._lib.sl_engine_get_stats(self._live(), ctypes.byref(stats)), "stats")
        return stats.as_dict()
