"""对外主入口：校验输入并串联时间轴、渲染、音频与封装。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence, Union

import numpy as np
from PIL import Image

from .audio import fit_audio, load_wav
from .encoder import encode
from .errors import ComposeError
from .frames import frame_plan, render_frame
from .timeline import SecondsLike, Timeline, build_timeline

PathLike = Union[str, Path]


def _load_image(path: PathLike) -> np.ndarray:
    path = Path(path)
    try:
        with Image.open(path) as im:
            if im.format != "PNG":
                raise ComposeError(f"仅支持 PNG 图片: {path}")
            if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
                raise ComposeError(f"图片不得包含透明通道: {path}")
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except ComposeError:
        raise
    except Exception as exc:
        raise ComposeError(f"无法读取图片 {path}: {exc}") from exc


def _frame_iter(images: Sequence[np.ndarray], timeline: Timeline) -> Iterator[np.ndarray]:
    for k in range(timeline.num_frames):
        yield render_frame(images, frame_plan(timeline, k))


def compose(
    images: Sequence[PathLike],
    displays: Sequence[SecondsLike],
    dissolves: Sequence[SecondsLike],
    music: PathLike,
    output: PathLike,
) -> Timeline:
    """合成无损 MOV 母版，返回使用的时间轴。

    images    按顺序排列的 PNG 图片路径（不含透明通道，尺寸一致）
    displays  每张图的显示时长（十进制秒，含其入场与出场转场）
    dissolves 每对相邻图片的溶解时长（十进制秒，可为零表示硬切）
    music     48kHz 双声道 16 位 PCM 的 WAV 路径
    output    输出 MOV 路径
    """
    if len(images) != len(displays):
        raise ComposeError(
            f"图片数量与显示时长数量不一致: {len(images)} != {len(displays)}"
        )
    timeline = build_timeline(displays, dissolves)
    frames = [_load_image(p) for p in images]
    height, width = frames[0].shape[:2]
    for i, f in enumerate(frames):
        if f.shape[:2] != (height, width):
            raise ComposeError(
                f"第 {i} 张图尺寸 {f.shape[1]}x{f.shape[0]} 与首张 {width}x{height} 不一致"
            )
    pcm = fit_audio(load_wav(music), timeline.audio_samples)
    encode(_frame_iter(frames, timeline), pcm, width, height, output)
    return timeline
