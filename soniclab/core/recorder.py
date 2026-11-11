"""Input capture that only copies blocks on the audio thread.

Assembly and disk IO happen after the stream has stopped, on a worker thread.
"""

import threading

import numpy as np
import sounddevice as sd

from soniclab.core.audio_engine import BLOCK_SIZE, OUTPUT_RATE


def input_devices():
    """Return (devices, error). Each device is (index, name, channel_count)."""
    try:
        listed = sd.query_devices()
    except Exception as exc:
        return [], str(exc)
    devices = []
    for index, device in enumerate(listed):
        channels = int(device.get("max_input_channels", 0))
        if channels > 0:
            devices.append((index, str(device.get("name", f"Input {index}")), channels))
    return devices, None


def assemble_blocks(blocks, channel, stereo: bool) -> np.ndarray:
    """Join captured blocks into a stereo float32 buffer."""
    if not blocks:
        return np.zeros((0, 2), dtype=np.float32)
    audio = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    if audio.ndim == 1:
        audio = audio[:, None]
    if stereo and audio.shape[1] >= 2:
        pair = audio[:, :2]
    else:
        channel = min(max(0, int(channel)), audio.shape[1] - 1)
        column = audio[:, channel : channel + 1]
        pair = np.repeat(column, 2, axis=1)
    return np.ascontiguousarray(pair, dtype=np.float32)


class Recorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._stream = None
        self._blocks = []
        self._result = None
        self._worker = None
        self.error = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self, device, channels: int, channel: int, stereo: bool) -> None:
        self.stop_stream()
        self.error = None
        with self._lock:
            self._blocks = []
            self._result = None
        channels = max(1, int(channels))

        def callback(indata, frames, time_info, status):
            if status:
                self.error = str(status)
            chunk = np.array(indata, dtype=np.float32, copy=True)
            with self._lock:
                self._blocks.append(chunk)

        self._stream = sd.InputStream(
            device=device,
            samplerate=OUTPUT_RATE,
            channels=channels,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        )
        self._stream.start()
        self._channel = int(channel)
        self._stereo = bool(stereo)

    def stop_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                self.error = str(exc)

    def finish_async(self, path: str) -> None:
        """Stop input, then write a WAV without blocking the caller."""
        self.stop_stream()
        with self._lock:
            blocks = self._blocks
            self._blocks = []
        channel = getattr(self, "_channel", 0)
        stereo = getattr(self, "_stereo", False)

        def work():
            try:
                audio = assemble_blocks(blocks, channel, stereo)
                if len(audio) == 0:
                    self._publish(("empty", path, audio))
                    return
                import soundfile as sf

                sf.write(path, audio, OUTPUT_RATE)
                self._publish(("ok", path, audio))
            except Exception as exc:
                self._publish(("error", str(exc), None))

        self._worker = threading.Thread(target=work, name="soniclab-record", daemon=True)
        self._worker.start()

    def poll(self):
        with self._lock:
            result, self._result = self._result, None
        return result

    def _publish(self, result) -> None:
        with self._lock:
            self._result = result
