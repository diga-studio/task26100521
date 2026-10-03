"""WAV music preparation: loop or truncate to the exact master sample count.

The source must be 48 kHz stereo 16-bit PCM.  Looping repeats the whole file
seamlessly from its first sample; no silence or fades are ever added.  All
arithmetic is on integer sample frames, so repeated loops cannot drift.
"""

from __future__ import annotations

import wave
from fractions import Fraction
from pathlib import Path
from typing import Union

from .timeline import FPS_DEN, FPS_NUM

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2


def master_sample_count(frame_count: int) -> int:
    """Samples covering the full video playback duration, rounded up.

    Equals ceil(frame_count * 1001 / 30000 * 48000): the last sample ends no
    earlier than the video end, overshooting by less than one sample period.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    exact = Fraction(frame_count * FPS_DEN * SAMPLE_RATE, FPS_NUM)
    return -(-exact.numerator // exact.denominator)


def _read_source(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"music must be {SAMPLE_RATE} Hz, got {wav.getframerate()}")
        if wav.getnchannels() != CHANNELS:
            raise ValueError(f"music must be {CHANNELS} channels, got {wav.getnchannels()}")
        if wav.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(
                f"music must be {SAMPLE_WIDTH * 8}-bit PCM, got {wav.getsampwidth() * 8}-bit"
            )
        if wav.getcomptype() != "NONE":
            raise ValueError(f"music must be uncompressed PCM, got {wav.getcomptype()}")
        data = wav.readframes(wav.getnframes())
    if not data:
        raise ValueError("music file contains no samples")
    return data


def prepare_music(source: Union[Path, str], samples: int, dest: Union[Path, str]) -> Path:
    """Loop (seamlessly) or truncate the source WAV to exactly ``samples`` frames."""
    if samples < 0:
        raise ValueError("samples must be non-negative")
    source = Path(source)
    dest = Path(dest)
    frame_bytes = CHANNELS * SAMPLE_WIDTH
    data = _read_source(source)
    src_frames = len(data) // frame_bytes

    repeats, remainder = divmod(samples, src_frames)
    out = bytearray()
    out += data * repeats
    out += data[: remainder * frame_bytes]

    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(out))
    return dest
