"""逐帧画面归属判定与 8 位 RGB 线性混合渲染。"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np

from .errors import ComposeError
from .timeline import FRAME_DURATION, Timeline


@dataclass(frozen=True)
class FramePlan:
    """单帧渲染计划。

    old_index/new_index 为参与混合的图片下标；weight 为新图权重。
    纯画面时 old_index == new_index 且 weight == 1。
    """

    old_index: int
    new_index: int
    weight: Fraction


def frame_plan(timeline: Timeline, k: int) -> FramePlan:
    """判定第 k 帧（取样时刻 k*1001/30000 秒）的画面归属与混合权重。

    归属规则：取起始时刻不大于 t 的最后一张图作为基准区间，
    交界时刻归属从该时刻开始的新区间；若 t 落在该图入场溶解
    区间内，则与前一张图按经过时间占比线性混合。
    """
    t = k * FRAME_DURATION
    starts = timeline.starts
    n = len(starts)
    base = n - 1
    for i in range(n - 1):
        if t < starts[i + 1]:
            base = i
            break
    if base > 0:
        in_dissolve = timeline.dissolves[base - 1]
        if in_dissolve > 0:
            elapsed = t - starts[base]
            if elapsed < in_dissolve:
                return FramePlan(base - 1, base, elapsed / in_dissolve)
    return FramePlan(base, base, Fraction(1))


def render_frame(images: Sequence[np.ndarray], plan: FramePlan) -> np.ndarray:
    """按计划渲染一帧：纯画面直接返回原图，否则逐像素线性混合。

    混合在 8 位 RGB 通道上以整数精确计算：
    out = round(old*(1-w) + new*w)，四舍五入（半数向上）。
    """
    old = images[plan.old_index]
    if plan.old_index == plan.new_index:
        return old
    new = images[plan.new_index]
    p, q = plan.weight.numerator, plan.weight.denominator
    if q >= (1 << 56):
        raise ComposeError("溶解权重分母过大，无法精确混合")
    old_u = old.astype(np.uint64)
    new_u = new.astype(np.uint64)
    mixed = (old_u * (q - p) + new_u * p + q // 2) // q
    return mixed.astype(np.uint8)
