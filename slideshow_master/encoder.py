"""调用本机 FFmpeg，将原始帧与 PCM 音频封装为无损 MOV 母版。"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from .errors import ComposeError
from .timeline import AUDIO_RATE, FRAME_RATE


def encode(
    frames_iter: Iterable[np.ndarray],
    audio_pcm: np.ndarray,
    width: int,
    height: int,
    output_path: Union[str, Path],
) -> None:
    """把 RGB24 帧序列与立体声 int16 PCM 封装为无损 MOV。

    视频使用 qtrle（QuickTime Animation，无损 RGB），音频使用
    pcm_s16le；帧通过 stdin 以 rawvideo 送入，避免中间文件损失。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    try:
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(audio_pcm.astype("<i2", copy=False).tobytes())
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-framerate", f"{FRAME_RATE.numerator}/{FRAME_RATE.denominator}",
            "-i", "pipe:0",
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "qtrle",
            "-pix_fmt", "rgb24",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame in frames_iter:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            output_path.unlink(missing_ok=True)
            raise ComposeError(
                "FFmpeg 封装失败:\n" + stderr.decode("utf-8", "replace")[-2000:]
            )
    finally:
        audio_path.unlink(missing_ok=True)
