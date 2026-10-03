"""slideshow_master 自动化测试。

运行：python3 -m pytest tests -v
需要本机可用的 ffmpeg 与 ffprobe。
"""
from __future__ import annotations

import json
import math
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slideshow_master import ComposeError, compose
from slideshow_master.timeline import AUDIO_RATE, FRAME_DURATION, build_timeline

W, H = 24, 16

COLORS = [
    (220, 10, 10),
    (10, 220, 10),
    (10, 10, 220),
    (220, 220, 10),
    (220, 10, 220),
]


def make_images(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for i in range(count):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(
            np.full((H, W, 3), COLORS[i % len(COLORS)], dtype=np.uint8)
        ).save(p)
        paths.append(p)
    return paths


def make_music(tmp_path: Path, num_samples: int) -> Path:
    """生成每个采样值都不同的 48k/立体声/16 位 WAV，便于检验循环拼接。"""
    p = tmp_path / "music.wav"
    left = (np.arange(num_samples, dtype=np.int64) * 7 % 30000 - 15000).astype("<i2")
    right = (np.arange(num_samples, dtype=np.int64) * 13 % 20000 - 10000).astype("<i2")
    pcm = np.stack([left, right], axis=1)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(pcm.tobytes())
    return p


def decode_video(path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(data) // (W * H * 3)
    return data[: n * W * H * 3].reshape(n, H, W, 3)


def decode_audio(path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True,
    )
    data = np.frombuffer(proc.stdout, dtype="<i2")
    return data[: len(data) // 2 * 2].reshape(-1, 2)


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, check=True, text=True,
    )
    return {s["codec_type"]: s for s in json.loads(proc.stdout)["streams"]}


def expected_frame(colors: list[np.ndarray], starts, displays, dissolves, k: int) -> np.ndarray:
    """独立参考实现：按全局取样时刻判定归属并线性混合。"""
    t = k * FRAME_DURATION
    base = len(starts) - 1
    for i in range(len(starts) - 1):
        if t < starts[i + 1]:
            base = i
            break
    if base > 0 and dissolves[base - 1] > 0:
        elapsed = t - starts[base]
        if elapsed < dissolves[base - 1]:
            w = elapsed / dissolves[base - 1]
            p, q = w.numerator, w.denominator
            old = colors[base - 1].astype(np.uint64)
            new = colors[base].astype(np.uint64)
            return ((old * (q - p) + new * p + q // 2) // q).astype(np.uint8)
    return colors[base]


def run_and_check(tmp_path, displays, dissolves, music_samples=48000):
    imgs = make_images(tmp_path, len(displays))
    music = make_music(tmp_path, music_samples)
    out = tmp_path / "master.mov"
    tl = compose(imgs, displays, dissolves, music, out)
    colors = [np.full((H, W, 3), COLORS[i % len(COLORS)], dtype=np.uint8)
              for i in range(len(displays))]
    return tl, out, colors


def test_decimal_accumulation_and_subframe_dissolve(tmp_path):
    """多段小数时长累积并夹有不足一帧的转场。"""
    displays = ["0.7", "1.1", "0.9", "1.3"]
    dissolves = ["0.02", "0.5", "0.03"]  # 0.02/0.03 秒均不足一帧
    tl, out, colors = run_and_check(tmp_path, displays, dissolves)

    assert tl.total == Fraction("3.45")
    assert tl.num_frames == math.ceil(Fraction("3.45") / FRAME_DURATION) == 104

    decoded = decode_video(out)
    assert decoded.shape[0] == tl.num_frames
    for k in range(tl.num_frames):
        want = expected_frame(colors, tl.starts, tl.displays, tl.dissolves, k)
        np.testing.assert_array_equal(decoded[k], want, err_msg=f"frame {k}")

    streams = probe(out)
    assert streams["video"]["codec_name"] == "qtrle"
    assert streams["video"]["pix_fmt"] == "rgb24"
    assert streams["video"]["r_frame_rate"] == "30000/1001"
    assert streams["audio"]["codec_name"] == "pcm_s16le"
    assert streams["audio"]["sample_rate"] == "48000"
    assert streams["audio"]["channels"] == 2


def test_touching_dissolves_and_hard_cut(tmp_path):
    """两个转场恰好相接（中间图无独占帧）以及零时长硬切。"""
    displays = ["1.0", "0.6", "0.5", "1.0"]
    dissolves = ["0.3", "0.3", "0"]  # 图1入场+出场=显示时长；图2->图3硬切
    tl, out, colors = run_and_check(tmp_path, displays, dissolves)

    assert tl.total == Fraction("2.5")
    assert tl.num_frames == 75

    decoded = decode_video(out)
    assert decoded.shape[0] == 75
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    for k in range(75):
        want = expected_frame(colors, tl.starts, tl.displays, tl.dissolves, k)
        np.testing.assert_array_equal(decoded[k], want, err_msg=f"frame {k}")
        assert not np.array_equal(decoded[k], blank), f"frame {k} 出现空帧"

    # 硬切边界：t = 1.5s，前后帧分别纯属于图2与图3
    boundary = Fraction("1.5")
    k_before = math.ceil(boundary / FRAME_DURATION) - 1
    k_after = math.ceil(boundary / FRAME_DURATION)
    assert k_before * FRAME_DURATION < boundary <= k_after * FRAME_DURATION
    np.testing.assert_array_equal(decoded[k_before], colors[2])
    np.testing.assert_array_equal(decoded[k_after], colors[3])


def test_audio_loops_without_drift(tmp_path):
    """短音乐多次循环：采样数精确、拼接无漂移、末尾覆盖视频终点。"""
    music_samples = 1000
    tl, out, _ = run_and_check(
        tmp_path, ["1.5", "2.0", "1.0"], ["0.5", "0.25"],
        music_samples=music_samples,
    )
    video_end = tl.num_frames * FRAME_DURATION
    assert tl.audio_samples == math.ceil(video_end * AUDIO_RATE)
    # 最后一个有效采样结束不早于视频结束，且超出不足一个采样周期
    assert tl.audio_samples >= video_end * AUDIO_RATE
    assert tl.audio_samples - 1 < video_end * AUDIO_RATE

    decoded = decode_audio(out)
    assert decoded.shape == (tl.audio_samples, 2)
    # 重新生成同源音乐作为参照
    left = (np.arange(music_samples, dtype=np.int64) * 7 % 30000 - 15000).astype("<i2")
    right = (np.arange(music_samples, dtype=np.int64) * 13 % 20000 - 10000).astype("<i2")
    src = np.stack([left, right], axis=1)
    want = np.tile(src, (tl.audio_samples // music_samples + 1, 1))[: tl.audio_samples]
    np.testing.assert_array_equal(decoded, want)


def test_audio_truncated_when_longer(tmp_path):
    """音乐长于需求时截断，不额外添加静音。"""
    tl, out, _ = run_and_check(
        tmp_path, ["0.2", "0.2"], ["0"], music_samples=48000,
    )
    decoded = decode_audio(out)
    assert decoded.shape == (tl.audio_samples, 2)
    left = (np.arange(48000, dtype=np.int64) * 7 % 30000 - 15000).astype("<i2")
    right = (np.arange(48000, dtype=np.int64) * 13 % 20000 - 10000).astype("<i2")
    src = np.stack([left, right], axis=1)
    np.testing.assert_array_equal(decoded, src[: tl.audio_samples])


def test_validation_errors(tmp_path):
    imgs = make_images(tmp_path, 2)
    music = make_music(tmp_path, 4800)
    out = tmp_path / "x.mov"

    with pytest.raises(ComposeError):  # 溶解时长数量不符
        compose(imgs, ["1", "1"], ["0.1", "0.1"], music, out)
    with pytest.raises(ComposeError):  # 入场+出场超过显示时长
        compose(imgs, ["0.2", "0.2"], ["0.3"], music, out)
    with pytest.raises(ComposeError):  # 负时长
        compose(imgs, ["1", "-0.5"], ["0"], music, out)
    with pytest.raises(ComposeError):  # 图片数与时长数不符
        compose(imgs, ["1"], ["0"], music, out)

    rgba = tmp_path / "rgba.png"
    Image.fromarray(np.zeros((H, W, 4), dtype=np.uint8)).save(rgba)
    with pytest.raises(ComposeError):  # 透明通道
        compose([rgba, imgs[0]], ["1", "1"], ["0"], music, out)

    bad = tmp_path / "bad.wav"
    with wave.open(str(bad), "wb") as wf:  # 44.1k 单声道
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(100, dtype="<i2").tobytes())
    with pytest.raises(ComposeError):
        compose(imgs, ["1", "1"], ["0"], bad, out)


def test_timeline_exactness():
    """十进制秒按分数精确累积，不产生二进制浮点漂移。"""
    tl = build_timeline(["0.1"] * 10, ["0.02"] * 9)
    assert tl.total == Fraction(1, 1) - Fraction(18, 100) == Fraction(82, 100)
    assert tl.starts[3] == Fraction("0.3") - Fraction("0.06")
