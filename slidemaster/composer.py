"""Render frames, mux with music via FFmpeg into a lossless MOV master.

Video: lossless QuickTime RLE (``qtrle``) with ``rgb24`` pixels — every frame
is exactly the blended 8-bit RGB result, with no trailing black or duplicate
frame.  Audio: 48 kHz stereo 16-bit PCM, looped/truncated by :mod:`.audio`.
Frames stream to FFmpeg over a pipe, so no intermediate files accumulate.
"""

from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
from PIL import Image

from .audio import prepare_music, master_sample_count
from .timeline import FPS_DEN, FPS_NUM, FrameKind, Timeline

NumberLike = Union[str, int, float, Fraction]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if "A" in im.getbands() or "transparency" in im.info:
            raise ValueError(f"{path}: images must not have an alpha channel")
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _blend(old: np.ndarray, new: np.ndarray, num: int, den: int) -> np.ndarray:
    """Per-pixel linear mix on 8-bit RGB, rounded to nearest (half up)."""
    acc = old.astype(np.uint32) * (den - num) + new.astype(np.uint32) * num
    acc += den // 2
    return (acc // den).astype(np.uint8)


def _run_ffmpeg(width: int, height: int, audio: Path, output: Path) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{FPS_NUM}/{FPS_DEN}",
        "-i", "pipe:0",
        "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "qtrle",
        "-pix_fmt", "rgb24",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )


def compose(
    images: Sequence[Union[str, Path]],
    durations: Sequence[NumberLike],
    dissolves: Sequence[NumberLike],
    music: Union[str, Path],
    output: Union[str, Path],
    *,
    work_dir: Union[str, Path, None] = None,
) -> Path:
    """Compose the slideshow master. Returns the output MOV path.

    ``images`` are ordered PNG paths without alpha; ``durations`` are display
    seconds per image (including its in/out transitions); ``dissolves`` are the
    dissolve seconds between each adjacent pair; ``music`` is a 48 kHz stereo
    16-bit PCM WAV.  ``work_dir`` holds the prepared audio temp file.
    """
    paths = [Path(p) for p in images]
    if not paths:
        raise ValueError("at least one image is required")
    timeline = Timeline(durations, dissolves)
    if len(paths) != len(timeline.durations):
        raise ValueError(
            f"{len(paths)} images but {len(timeline.durations)} display durations"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[int, np.ndarray] = {}

    def frame_pixels(index: int) -> np.ndarray:
        if index not in cache:
            cache[index] = _load_rgb(paths[index])
        return cache[index]

    first = frame_pixels(0)
    height, width = first.shape[:2]
    for i, p in enumerate(paths):
        arr = frame_pixels(i)
        if arr.shape != first.shape:
            raise ValueError(
                f"image #{i} ({p}) is {arr.shape[1]}x{arr.shape[0]}, "
                f"expected {width}x{height}"
            )

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        audio = prepare_music(
            music, master_sample_count(timeline.frame_count), Path(tmp) / "music.wav"
        )
        proc = _run_ffmpeg(width, height, audio, output)
        assert proc.stdin is not None
        try:
            for plan in timeline.plans():
                if plan.kind is FrameKind.SOLID:
                    pixels = frame_pixels(plan.image_index)
                else:
                    pixels = _blend(
                        frame_pixels(plan.image_index - 1),
                        frame_pixels(plan.image_index),
                        plan.weight_num,
                        plan.weight_den,
                    )
                proc.stdin.write(pixels.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.wait() != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {proc.returncode}:\n"
                + stderr.decode(errors="replace")
            )
    return output
