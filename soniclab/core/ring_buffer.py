"""Fixed stereo ring shared by the audio thread and the UI.

The audio callback only copies into storage allocated ahead of time.
Reading copies out under the same lock and never runs inside the callback.
"""

import threading

import numpy as np


class StereoRing:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.data = np.zeros((self.capacity, 2), dtype=np.float32)
        self.write = 0
        self.filled = 0
        self._lock = threading.Lock()

    def push(self, frames) -> None:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 2:
            return
        count = int(frames.shape[0])
        if count <= 0:
            return
        if count >= self.capacity:
            frames = frames[-self.capacity :]
            count = self.capacity
        with self._lock:
            first = min(count, self.capacity - self.write)
            self.data[self.write : self.write + first] = frames[:first]
            rest = count - first
            if rest:
                self.data[:rest] = frames[first:]
            self.write = (self.write + count) % self.capacity
            self.filled = min(self.capacity, self.filled + count)

    def latest(self, count: int) -> np.ndarray:
        with self._lock:
            count = min(int(count), self.filled, self.capacity)
            if count <= 0:
                return np.zeros((0, 2), dtype=np.float32)
            start = (self.write - count) % self.capacity
            out = np.empty((count, 2), dtype=np.float32)
            first = min(count, self.capacity - start)
            out[:first] = self.data[start : start + first]
            if first < count:
                out[first:] = self.data[: count - first]
            return out

    def clear(self) -> None:
        with self._lock:
            self.data.fill(0)
            self.write = 0
            self.filled = 0
