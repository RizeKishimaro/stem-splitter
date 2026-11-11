"""Audition one audio file without putting it in the project.

The Stem Splitter uses this to play the source song and each stem. It opens
its own output stream, separate from the project's ``AudioEngine``, so it
needs no change to the engine and cannot disturb the arrangement.

Threads
    ``play`` and ``stop`` run on the UI thread. The stream callback only
    copies from a buffer loaded before the stream starts and moves one
    integer; it takes no lock and allocates nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


class PreviewPlayer:
    def __init__(self, stream_factory=None):
        # ``stream_factory(samplerate, channels, callback)`` returns an object
        # with start/stop/close. Tests pass a fake; the default is PortAudio.
        self._stream_factory = stream_factory
        self._stream = None
        self._audio = np.zeros((0, 2), dtype=np.float32)
        self._rate = 44100
        self._position = 0
        self._finished = True
        self._path: str | None = None

    @property
    def path(self) -> str | None:
        """The file being played, or None."""
        return self._path if self.playing else None

    @property
    def playing(self) -> bool:
        return self._stream is not None and not self._finished

    @property
    def position_seconds(self) -> float:
        return self._position / self._rate if self._stream is not None else 0.0

    def play(self, path) -> str:
        """Start playing ``path`` from its beginning. Returns "" or an error."""
        self.stop()
        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            return f"Could not read {Path(path).name}: {exc}"
        if audio.shape[0] == 0:
            return f"{Path(path).name} is empty"
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        self._audio = np.ascontiguousarray(audio[:, :2])
        self._rate = int(rate)
        self._position = 0
        self._finished = False
        try:
            factory = self._stream_factory or _portaudio_stream
            self._stream = factory(self._rate, 2, self._callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self._finished = True
            return f"Could not open the audio output: {exc}"
        self._path = str(path)
        return ""

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._finished = True
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def _callback(self, outdata, frames, _time, _status) -> None:
        start = self._position
        chunk = self._audio[start : start + frames]
        count = chunk.shape[0]
        outdata[:count] = chunk
        if count < frames:
            outdata[count:] = 0.0
            self._finished = True
        self._position = start + count


def _portaudio_stream(samplerate: int, channels: int, callback):
    import sounddevice as sd

    return sd.OutputStream(samplerate=samplerate, channels=channels, dtype="float32", callback=callback)
