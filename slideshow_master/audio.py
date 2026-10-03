"""音轨处理：读取 WAV，按帧数循环或截断到精确采样数。"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Union

import numpy as np

from .errors import ComposeError
from .timeline import AUDIO_RATE


def load_wav(path: Union[str, Path]) -> np.ndarray:
    """读取 48kHz 双声道 16 位 PCM 的 WAV，返回 (n, 2) 的 int16 数组。"""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getframerate() != AUDIO_RATE:
                raise ComposeError(
                    f"音乐采样率必须为 {AUDIO_RATE}Hz: {wf.getframerate()}"
                )
            if wf.getnchannels() != 2:
                raise ComposeError(f"音乐必须为双声道: {wf.getnchannels()}")
            if wf.getsampwidth() != 2:
                raise ComposeError(f"音乐必须为 16 位 PCM: {wf.getsampwidth() * 8} 位")
            if wf.getcomptype() != "NONE":
                raise ComposeError(f"音乐必须为未压缩 PCM: {wf.getcomptype()}")
            frames = wf.getnframes()
            raw = wf.readframes(frames)
    except wave.Error as exc:
        raise ComposeError(f"无法读取 WAV 文件 {path}: {exc}") from exc
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    if pcm.shape[0] == 0:
        raise ComposeError("音乐文件不含任何采样")
    return pcm.copy()


def fit_audio(pcm: np.ndarray, num_samples: int) -> np.ndarray:
    """把音轨调整为精确的采样数：短则首尾连续循环，长则截断。"""
    if num_samples <= 0:
        raise ComposeError("音轨采样数必须为正")
    n = pcm.shape[0]
    if n == num_samples:
        return pcm
    if n > num_samples:
        return pcm[:num_samples].copy()
    repeats = (num_samples + n - 1) // n
    return np.tile(pcm, (repeats, 1))[:num_samples].copy()
