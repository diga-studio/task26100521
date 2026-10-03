import math
from fractions import Fraction

import pytest

from slidemaster.timeline import FRAME_DURATION, FrameKind, Timeline


def test_total_duration_is_displays_minus_dissolves():
    tl = Timeline(["1.5", "2", "1"], ["0.5", "0.25"])
    assert tl.total_duration == Fraction(15, 10) + 2 + 1 - Fraction(3, 4)
    assert tl.total_duration == Fraction(15, 4)


def test_frame_count_counts_instants_strictly_before_end():
    # 3.75 s -> ceil(3.75 * 30000 / 1001) = ceil(112.387...) = 113
    tl = Timeline(["1.5", "2", "1"], ["0.5", "0.25"])
    assert tl.frame_count == 113
    # Exact multiple of the frame period: the frame landing exactly on the
    # end instant is excluded.
    exact = Timeline(["1001"], [])  # 1001 s = 30000 frames exactly
    assert exact.frame_count == 30000
    assert exact.frame_plan(29999).image_index == 0
    with pytest.raises(IndexError):
        exact.frame_plan(30000)


def test_decimal_accumulation_has_no_drift():
    # 0.1 cannot be represented exactly in binary; rational math must keep
    # the logical end exact after many segments.
    n = 30
    tl = Timeline(["0.1"] * n, ["0.01"] * (n - 1))
    assert tl.total_duration == Fraction(30) / 10 - Fraction(29) / 100
    assert tl.total_duration == Fraction(271, 100)
    assert tl.frame_count == math.ceil(Fraction(271, 100) * 30000 / 1001)


def test_boundary_belongs_to_interval_starting_there():
    # Hard cut placed exactly on a frame instant: that frame shows image 1.
    cut = 10 * FRAME_DURATION
    tl = Timeline([cut, "1"], ["0"])
    assert tl.frame_plan(10).image_index == 1
    assert tl.frame_plan(9).image_index == 0


def test_blend_weights_and_hard_cut():
    # Transition starts exactly on frame 15's instant and lasts 0.5 s.
    start = 15 * FRAME_DURATION
    tl = Timeline([start + Fraction(1, 2), "1"], ["0.5"])
    plans = list(tl.plans())
    blends = [(k, p) for k, p in enumerate(plans) if p.kind is FrameKind.BLEND]
    assert blends, "expected blend frames"
    first_k, first = blends[0]
    assert first_k == 15
    assert (first.weight_num, first.weight_den) == (0, 1)
    for k, p in blends:
        t = k * FRAME_DURATION
        w = (t - start) / Fraction(1, 2)
        assert p.image_index == 1
        assert Fraction(p.weight_num, p.weight_den) == w
        assert 0 <= w < 1
    # The first frame at or after the transition end is fully the new image.
    after = [p for k, p in enumerate(plans) if k * FRAME_DURATION >= start + Fraction(1, 2)]
    assert after[0].kind is FrameKind.SOLID
    assert after[0].image_index == 1


def test_sub_frame_dissolve_still_uses_global_clock():
    # Dissolve shorter than one frame period: ownership decided purely by
    # the global instant, never by accumulating per-segment rounding.
    tl = Timeline(["0.5", "0.5"], ["0.01"])
    assert tl.frame_count == math.ceil(Fraction(99, 100) * 30000 / 1001)
    for k, plan in enumerate(tl.plans()):
        t = k * FRAME_DURATION
        if t < Fraction(49, 100):
            assert plan.kind is FrameKind.SOLID and plan.image_index == 0
        elif t < Fraction(1, 2):
            assert plan.kind is FrameKind.BLEND and plan.image_index == 1
        else:
            assert plan.kind is FrameKind.SOLID and plan.image_index == 1


def test_adjacent_transitions_leave_no_exclusive_frames():
    # Image 1's in+out dissolves exactly equal its display duration.
    tl = Timeline(["1", "0.6", "1"], ["0.3", "0.3"])
    assert tl.total_duration == Fraction(2)
    kinds = [p.kind for p in tl.plans()]
    assert FrameKind.BLEND in kinds
    # No solid frame of image 1 exists, and blends only ever pair neighbours.
    for k, p in enumerate(tl.plans()):
        t = k * FRAME_DURATION
        if t < Fraction(7, 10):
            assert (p.kind, p.image_index) == (FrameKind.SOLID, 0)
        elif t < Fraction(1):
            assert (p.kind, p.image_index) == (FrameKind.BLEND, 1)
        elif t < Fraction(13, 10):
            assert (p.kind, p.image_index) == (FrameKind.BLEND, 2)
        else:
            assert (p.kind, p.image_index) == (FrameKind.SOLID, 2)


def test_validation():
    with pytest.raises(ValueError):
        Timeline([], [])
    with pytest.raises(ValueError):
        Timeline(["1", "1"], [])  # missing dissolve
    with pytest.raises(ValueError):
        Timeline(["1", "1"], ["-0.1"])
    with pytest.raises(ValueError):
        Timeline(["0", "1"], ["0"])
    with pytest.raises(ValueError):
        Timeline(["1", "0.4", "1"], ["0.3", "0.3"])  # in+out > display
    # Exactly equal is allowed.
    Timeline(["1", "0.6", "1"], ["0.3", "0.3"])
