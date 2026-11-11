from collections import Counter

import numpy as np

from madmom.features.downbeats import (
    RNNDownBeatProcessor,
    DBNDownBeatTrackingProcessor,
)

audio = "/home/rizekishimaro/Music/Burma House [2HcLKKpY4Dk].mp3"

print("Loading Madmom neural network...")
activations = RNNDownBeatProcessor()(audio)

print("Detecting beats and downbeats...")
tracker = DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4],
    fps=100,
)
beats = tracker(activations)

timestamps = beats[:, 0]
beat_numbers = beats[:, 1].astype(int)

intervals = np.diff(timestamps)
median_interval = np.median(intervals)

print("\n========== ANALYSIS RESULTS ==========")
print(f"Total detected beats: {len(beats)}")
print(f"Detected downbeats: {np.sum(beat_numbers == 1)}")
print(f"Median beat interval: {median_interval:.3f} seconds")
print(f"Estimated tempo: {60 / median_interval:.2f} BPM")
print(f"Shortest interval: {intervals.min():.3f} seconds")
print(f"Longest interval: {intervals.max():.3f} seconds")

print("\nBeat-position distribution:")
for number, count in sorted(Counter(beat_numbers).items()):
    print(f"Beat {number}: {count}")

print("\nFirst 12 events:")
for timestamp, number in beats[:12]:
    label = "DOWNBEAT" if int(number) == 1 else "Beat"
    print(f"{timestamp:8.3f}s | Beat {int(number)} | {label}")

print("\nLast 12 events:")
for timestamp, number in beats[-12:]:
    label = "DOWNBEAT" if int(number) == 1 else "Beat"
    print(f"{timestamp:8.3f}s | Beat {int(number)} | {label}")

print("\n======================================")
