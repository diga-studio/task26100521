"""时间轴计算：全部使用分数精确运算，避免小数累积漂移。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import List, Sequence, Union

from .errors import ComposeError

FRAME_RATE = Fraction(30000, 1001)
FRAME_DURATION = Fraction(1001, 30000)
AUDIO_RATE = 48000

SecondsLike = Union[int, float, str, Decimal, Fraction]


@dataclass(frozen=True)
class Timeline:
    """整片时间轴。

    starts[i]    第 i 张图的全局起始时刻（含入场转场起点）
    displays[i]  第 i 张图的显示时长（含其入场与出场转场）
    dissolves[i] 第 i 张与第 i+1 张之间的溶解时长
    total        整片逻辑时长 = 显示时长之和 - 溶解时长之和
    num_frames   输出帧数：取样时刻严格早于 total 的帧数
    audio_samples 音轨采样数 = ceil(num_frames * FRAME_DURATION * 48000)
    """

    starts: List[Fraction]
    displays: List[Fraction]
    dissolves: List[Fraction]
    total: Fraction
    num_frames: int
    audio_samples: int


def to_fraction(value: SecondsLike) -> Fraction:
    """把十进制秒精确转换为分数，拒绝负值与非有限值。"""
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, bool):
        raise ComposeError(f"非法时长: {value!r}")
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, (float, str, Decimal)):
        try:
            result = Fraction(str(value))
        except (ValueError, ZeroDivisionError) as exc:
            raise ComposeError(f"非法时长: {value!r}") from exc
    else:
        raise ComposeError(f"非法时长类型: {type(value).__name__}")
    if result < 0:
        raise ComposeError(f"时长不得为负: {value!r}")
    return result


def build_timeline(
    displays: Sequence[SecondsLike],
    dissolves: Sequence[SecondsLike],
) -> Timeline:
    """根据显示时长与溶解时长构建精确时间轴并做合法性校验。"""
    if not displays:
        raise ComposeError("至少需要一张图片")
    d = [to_fraction(v) for v in displays]
    x = [to_fraction(v) for v in dissolves]
    if len(x) != len(d) - 1:
        raise ComposeError(
            f"溶解时长数量应为图片数减一: {len(x)} != {len(d) - 1}"
        )
    for i, di in enumerate(d):
        if di <= 0:
            raise ComposeError(f"第 {i} 张图显示时长必须为正: {di}")
    for i in range(len(d)):
        in_x = x[i - 1] if i > 0 else Fraction(0)
        out_x = x[i] if i < len(x) else Fraction(0)
        if in_x + out_x > d[i]:
            raise ComposeError(
                f"第 {i} 张图入场与出场溶解时长之和 {in_x + out_x} 超过其显示时长 {d[i]}"
            )
    starts = [Fraction(0)]
    for i in range(len(d) - 1):
        starts.append(starts[-1] + d[i] - x[i])
    total = starts[-1] + d[-1]
    num_frames = math.ceil(total / FRAME_DURATION)
    if num_frames <= 0:
        raise ComposeError("整片逻辑时长不足一帧")
    audio_samples = math.ceil(num_frames * FRAME_DURATION * AUDIO_RATE)
    return Timeline(
        starts=starts,
        displays=d,
        dissolves=x,
        total=total,
        num_frames=num_frames,
        audio_samples=audio_samples,
    )
