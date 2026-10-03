import json
import math
import struct
import subprocess
import wave
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

from slidemaster import Timeline, compose
from slidemaster.audio import master_sample_count
from slidemaster.timeline import FRAME_DURATION, FrameKind

W, H = 64, 48
COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (220, 220, 40)]


def _make_images(tmp_path, count):
    paths = []
    for i in range(count):
        p = tmp_path / f"im{i}.png"
        Image.fromarray(np.full((H, W, 3), COLORS[i % len(COLORS)], np.uint8)).save(p)
        paths.append(p)
    return paths


def _make_music(tmp_path, frames=4800):
    p = tmp_path / "music.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(
            b"".join(struct.pack("<hh", (i * 13) % 20000 - 10000, (i * 7) % 16000 - 8000)
                     for i in range(frames))
        )
    return p


def _decode_video(mov):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mov), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    frame_bytes = W * H * 3
    assert len(raw) % frame_bytes == 0
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def _decode_audio(mov):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mov), "-map", "0:a:0",
         "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
        check=True, capture_output=True,
    ).stdout


def _probe(mov):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(mov)],
        check=True, capture_output=True,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _expected_frame(timeline, k, arrays):
    plan = timeline.frame_plan(k)
    if plan.kind is FrameKind.SOLID:
        return arrays[plan.image_index]
    old = arrays[plan.image_index - 1].astype(np.uint32)
    new = arrays[plan.image_index].astype(np.uint32)
    num, den = plan.weight_num, plan.weight_den
    return ((old * (den - num) + new * num + den // 2) // den).astype(np.uint8)


def test_decimal_accumulation_with_subframe_dissolves(tmp_path):
    images = _make_images(tmp_path, 4)
    music = _make_music(tmp_path)
    durations = ["0.7", "1.1", "0.9", "0.6"]
    dissolves = ["0.0333", "0.4", "0.0167"]  # two are shorter than one frame
    out = compose(images, durations, dissolves, music, tmp_path / "master.mov")

    timeline = Timeline(durations, dissolves)
    assert timeline.total_duration == Fraction(57, 20)
    expected_frames = math.ceil(Fraction(57, 20) * 30000 / 1001)

    streams = _probe(out)
    assert streams["video"]["codec_name"] == "qtrle"
    assert streams["video"]["pix_fmt"] == "rgb24"
    assert int(streams["video"]["nb_frames"]) == expected_frames
    assert streams["audio"]["codec_name"] == "pcm_s16le"
    assert streams["audio"]["sample_rate"] == "48000"
    assert streams["audio"]["channels"] == 2

    decoded = _decode_video(out)
    assert decoded.shape[0] == expected_frames
    arrays = [np.full((H, W, 3), COLORS[i], np.uint8) for i in range(4)]
    for k in range(expected_frames):
        np.testing.assert_array_equal(
            decoded[k], _expected_frame(timeline, k, arrays), err_msg=f"frame {k}"
        )
    # Last frame is the final image, not a black or repeated-transition frame.
    assert timeline.frame_plan(expected_frames - 1).kind is FrameKind.SOLID
    np.testing.assert_array_equal(decoded[-1], arrays[3])

    # Audio: exact sample count and seamless looping of the source.
    expected_samples = master_sample_count(expected_frames)
    audio = _decode_audio(out)
    assert len(audio) == expected_samples * 4
    with wave.open(str(music), "rb") as w:
        src = w.readframes(w.getnframes())
    loops, rem = divmod(expected_samples * 4, len(src))
    assert audio == src * loops + src[:rem]


def test_adjacent_transitions_and_hard_cut_have_no_gaps(tmp_path):
    images = _make_images(tmp_path, 4)
    music = _make_music(tmp_path)
    # Image 1: in+out dissolves exactly fill its display (no exclusive frames).
    # Image 2 -> 3: zero dissolve, hard cut.
    durations = ["1", "0.6", "1", "0.8"]
    dissolves = ["0.3", "0.3", "0"]
    out = compose(images, durations, dissolves, music, tmp_path / "master.mov")

    timeline = Timeline(durations, dissolves)
    decoded = _decode_video(out)
    assert decoded.shape[0] == timeline.frame_count
    arrays = [np.full((H, W, 3), COLORS[i], np.uint8) for i in range(4)]

    black = np.zeros((H, W, 3), np.uint8)
    for k in range(timeline.frame_count):
        frame = decoded[k]
        assert not np.array_equal(frame, black), f"frame {k} is blank"
        np.testing.assert_array_equal(
            frame, _expected_frame(timeline, k, arrays), err_msg=f"frame {k}"
        )
        # Every frame is either a pure source image or a mix of exactly the
        # two adjacent images involved in one transition (never three).
        plan = timeline.frame_plan(k)
        involved = {plan.image_index} if plan.kind is FrameKind.SOLID else {
            plan.image_index - 1, plan.image_index
        }
        assert len(involved) <= 2

    # Hard cut between image 2 and 3 at t = 2.0 s (image 3 starts at
    # 0.7 + 0.6 - 0.3 + 1 - 0): frames strictly before are pure image 2,
    # frames at/after are pure image 3.
    cut = Fraction(2)
    for k in range(timeline.frame_count):
        t = k * FRAME_DURATION
        if abs(float(t - cut)) < 0.1:
            expected = arrays[2] if t < cut else arrays[3]
            np.testing.assert_array_equal(decoded[k], expected, err_msg=f"cut frame {k}")


def test_compose_validates_inputs(tmp_path):
    images = _make_images(tmp_path, 2)
    music = _make_music(tmp_path)
    with pytest.raises(ValueError):
        compose(images, ["1"], ["0"], music, tmp_path / "x.mov")  # count mismatch
    with pytest.raises(ValueError):
        compose(images, ["1", "1"], ["2"], music, tmp_path / "x.mov")  # dissolve > display
    rgba = tmp_path / "rgba.png"
    Image.fromarray(np.zeros((H, W, 4), np.uint8)).save(rgba)
    with pytest.raises(ValueError):
        compose([rgba, images[0]], ["1", "1"], ["0"], music, tmp_path / "x.mov")
