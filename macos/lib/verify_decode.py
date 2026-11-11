#!/usr/bin/env python3
"""Prove this Python can decode a WAV without a TorchAudio backend error.

Run it with the Demucs environment. SoundFile is required. TorchAudio is
checked only when it is installed. TorchAudio 2.9 and later must have
TorchCodec; older TorchAudio uses SoundFile and must not be upgraded here.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def click(path: Path) -> None:
    rate = 44100
    count = rate * 2
    frames = bytearray(count * 2)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/soniclab-decode-check.wav")
    click(wav)
    try:
        import soundfile as sf
    except Exception as exc:
        return fail(f"SoundFile is not installed in the stem-splitting environment: {exc}")
    try:
        audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    except Exception as exc:
        return fail(f"SoundFile could not read a WAV file: {exc}")
    if audio.ndim != 2 or audio.shape[0] < rate or rate != 44100:
        return fail("SoundFile read the WAV, but the audio was empty or the sample rate was wrong.")

    try:
        import torchaudio
    except Exception:
        print(f"SONICLAB_DECODE_OK soundfile frames={audio.shape[0]} channels={audio.shape[1]} torchaudio=absent")
        return 0

    version = str(getattr(torchaudio, "__version__", "0"))
    major_minor = version.split("+", 1)[0].split(".")
    try:
        major = int(major_minor[0])
        minor = int(major_minor[1]) if len(major_minor) > 1 else 0
    except ValueError:
        major, minor = 0, 0
    if (major, minor) >= (2, 9):
        try:
            import torchcodec  # noqa: F401
        except Exception as exc:
            return fail(
                "TorchAudio "
                f"{version} needs TorchCodec to open audio, and TorchCodec is not installed: {exc}"
            )
    try:
        loaded, loaded_rate = torchaudio.load(str(wav))
    except Exception as exc:
        text = str(exc).lower()
        if "appropriate backend" in text or "torchcodec" in text:
            return fail(
                "TorchAudio could not find a decoder for a WAV file. "
                "SoundFile is installed; this TorchAudio build is not using it. "
                f"The decoder said: {exc}"
            )
        return fail(f"TorchAudio could not read a WAV file: {exc}")
    if int(loaded_rate) != 44100 or loaded.numel() < 44100:
        return fail("TorchAudio read the WAV, but the audio was empty or the sample rate was wrong.")
    print(
        f"SONICLAB_DECODE_OK soundfile frames={audio.shape[0]} "
        f"torchaudio={version} torch_frames={int(loaded.shape[-1])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
