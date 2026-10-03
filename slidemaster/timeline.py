"""Exact rational timeline: frame ownership and blend weights.

All times are exact :class:`fractions.Fraction` seconds so decimal durations
accumulate without drift.  Frame ``k`` samples the global timeline at
``k * 1001 / 30000`` seconds; every interval is half-open ``[start, end)`` so a
boundary instant belongs to the interval that begins there.
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Iterator, List, Sequence, Tuple, Union

Number = Union[str, int, float, Decimal, Fraction]

FPS_NUM = 30000
FPS_DEN = 1001
FRAME_DURATION = Fraction(FPS_DEN, FPS_NUM)


class FrameKind(enum.Enum):
    SOLID = "solid"
    BLEND = "blend"


@dataclass(frozen=True)
class FramePlan:
    """Render plan for one output frame.

    For ``SOLID`` frames ``image_index`` is the image to show.  For ``BLEND``
    frames ``image_index`` is the incoming (new) image, the outgoing image is
    ``image_index - 1``, and the new image's weight is
    ``weight_num / weight_den`` in ``[0, 1)``.
    """

    kind: FrameKind
    image_index: int
    weight_num: int = 0
    weight_den: int = 1


def _to_fraction(value: Number) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("duration must be a number, not bool")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"unsupported duration type: {type(value)!r}")


class Timeline:
    """Global timeline from per-image display durations and pairwise dissolves."""

    def __init__(self, durations: Sequence[Number], dissolves: Sequence[Number]) -> None:
        if not durations:
            raise ValueError("at least one image duration is required")
        d = [_to_fraction(v) for v in durations]
        x = [_to_fraction(v) for v in dissolves]
        if len(x) != len(d) - 1:
            raise ValueError(
                f"expected {len(d) - 1} dissolve durations, got {len(x)}"
            )
        for i, di in enumerate(d):
            if di <= 0:
                raise ValueError(f"display duration #{i} must be positive, got {di}")
        for i, xi in enumerate(x):
            if xi < 0:
                raise ValueError(f"dissolve duration #{i} must be non-negative, got {xi}")
        for i in range(len(d)):
            in_x = x[i - 1] if i > 0 else Fraction(0)
            out_x = x[i] if i < len(x) else Fraction(0)
            if in_x + out_x > d[i]:
                raise ValueError(
                    f"image #{i}: in+out dissolves ({in_x} + {out_x}) exceed "
                    f"display duration {d[i]}"
                )

        self._durations = d
        self._dissolves = x

        starts = [Fraction(0)]
        for i in range(len(d) - 1):
            starts.append(starts[-1] + d[i] - x[i])
        self._starts = starts
        self._total = starts[-1] + d[-1]
        if self._total <= 0:
            raise ValueError("total logical duration must be positive")

        # Contiguous half-open segments tiling [0, total): solid_i then blend_i.
        segments: List[Tuple[Fraction, Fraction, FrameKind, int]] = []
        for i in range(len(d)):
            solid_begin = starts[i] + (x[i - 1] if i > 0 else Fraction(0))
            solid_end = starts[i + 1] if i < len(x) else self._total
            segments.append((solid_begin, solid_end, FrameKind.SOLID, i))
            if i < len(x) and x[i] > 0:
                segments.append((starts[i + 1], starts[i + 1] + x[i], FrameKind.BLEND, i + 1))
        self._segments = segments
        self._segment_starts = [s[0] for s in segments]

    @property
    def durations(self) -> Tuple[Fraction, ...]:
        return tuple(self._durations)

    @property
    def dissolves(self) -> Tuple[Fraction, ...]:
        return tuple(self._dissolves)

    @property
    def image_starts(self) -> Tuple[Fraction, ...]:
        """Global start instant of each image (including its incoming dissolve)."""
        return tuple(self._starts)

    @property
    def total_duration(self) -> Fraction:
        """Logical duration: sum(displays) - sum(dissolves)."""
        return self._total

    @property
    def frame_count(self) -> int:
        """Number of frames whose sample instant is strictly before the end."""
        return -(-self._total * FPS_NUM // FPS_DEN) if self._total > 0 else 0

    def frame_plan(self, k: int) -> FramePlan:
        if not 0 <= k < self.frame_count:
            raise IndexError(f"frame {k} out of range [0, {self.frame_count})")
        t = k * FRAME_DURATION
        idx = bisect_right(self._segment_starts, t) - 1
        start, _end, kind, image_index = self._segments[idx]
        if kind is FrameKind.SOLID:
            return FramePlan(FrameKind.SOLID, image_index)
        x = self._dissolves[image_index - 1]
        w = (t - start) / x
        return FramePlan(FrameKind.BLEND, image_index, w.numerator, w.denominator)

    def plans(self) -> Iterator[FramePlan]:
        for k in range(self.frame_count):
            yield self.frame_plan(k)
