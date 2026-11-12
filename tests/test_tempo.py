"""Tempo change without a pitch change (soniclab/core/tempo.py)."""

import time
import unittest

import numpy as np

from soniclab.core.tempo import stretch, tempo_factor

RATE = 44100


def peak_frequency(signal, rate=RATE):
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))
    index = int(np.argmax(spectrum))
    # Parabolic interpolation around the peak bin.
    a, b, c = spectrum[index - 1 : index + 2]
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    return (index + offset) * rate / len(signal)


def clicks(bpm, seconds, rate=RATE):
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    period = 60.0 / bpm
    times = np.arange(0.5, seconds - 0.2, period)
    for t in times:
        at = int(t * rate)
        burst = np.exp(-np.arange(800) / 120.0) * np.sin(2 * np.pi * 1500 * np.arange(800) / rate)
        audio[at : at + 800] += burst.astype(np.float32)
    return audio, times


def onset_times(audio, rate=RATE):
    envelope = np.abs(audio)
    threshold = 0.3 * envelope.max()
    above = envelope > threshold
    starts = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    found = []
    for s in starts:
        if not found or s - found[-1] > rate * 0.1:
            found.append(s)
    return np.array(found) / rate


class TempoTests(unittest.TestCase):
    def test_factor(self):
        self.assertAlmostEqual(tempo_factor(120, 60), 2.0)
        self.assertAlmostEqual(tempo_factor(100, 125), 0.8)
        with self.assertRaises(ValueError):
            tempo_factor(0, 120)

    def test_length_follows_the_factor_exactly(self):
        audio = np.random.default_rng(1).standard_normal((RATE * 2 + 37, 2)).astype(np.float32) * 0.1
        for factor in (0.8, 1.25, 2.0):
            out = stretch(audio, factor)
            self.assertEqual(out.shape, (int(round(len(audio) * factor)), 2))
            self.assertEqual(out.dtype, np.float32)
        self.assertEqual(stretch(audio[:, 0], 1.5).shape, (int(round(len(audio) * 1.5)),))
        self.assertTrue(np.array_equal(stretch(audio, 1.0), audio))

    def test_pitch_is_kept(self):
        t = np.arange(RATE * 2) / RATE
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for factor in (0.8, 1.25):
            out = stretch(tone, factor)
            middle = out[len(out) // 4 : len(out) // 4 + 16384]
            self.assertAlmostEqual(peak_frequency(middle), 440.0, delta=2.0)
            # Level is kept too (no hollow, quiet result).
            self.assertAlmostEqual(float(np.sqrt(np.mean(middle**2))), 0.5 / np.sqrt(2), delta=0.05)

    def test_beats_move_with_the_tempo(self):
        audio, times = clicks(120, 8)
        out = stretch(audio, tempo_factor(120, 100))  # slower: 1.2 x longer
        found = onset_times(out)
        expected = times * 1.2
        self.assertEqual(len(found), len(expected))
        self.assertLess(float(np.max(np.abs(found - expected))), 0.02)

    def test_drum_hits_stay_sharp_and_do_not_overshoot(self):
        # Short hits, one bar louder. A plain phase vocoder smears them and
        # pushes some peaks past the input's.
        t = np.arange(RATE * 16)
        drums = np.zeros(len(t), dtype=np.float32)
        for beat in np.arange(0.0, 16.0, 0.5):
            at = int(beat * RATE)
            drums[at : at + 2000] += (np.exp(-np.arange(2000) / 300.0) * (0.8 if 4 <= beat < 6 else 0.3)).astype(np.float32)
        for factor in (0.8, 1.2):
            out = stretch(drums, factor)
            self.assertLessEqual(float(np.abs(out).max()), 0.8 * 1.05)
            loud = float(np.mean(out[int(4 * factor * RATE) : int(6 * factor * RATE)] ** 2))
            quiet = float(np.mean(out[int(8 * factor * RATE) : int(10 * factor * RATE)] ** 2))
            self.assertGreater(loud / quiet, 4.5)  # 7.1 in the original

    def test_a_song_length_track_is_fast_enough(self):
        audio = np.random.default_rng(2).standard_normal((RATE * 30, 2)).astype(np.float32) * 0.1
        started = time.perf_counter()
        seen = []
        stretch(audio, 0.9, progress=seen.append)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 30.0)  # 30 s of stereo audio; a worker thread does it
        self.assertTrue(seen)
        print(f"\n  stretching 30 s of stereo took {elapsed:.1f} s")


if __name__ == "__main__":
    unittest.main()
