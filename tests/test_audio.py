import math
import struct
import wave
from fractions import Fraction

import pytest

from slidemaster.audio import master_sample_count, prepare_music


def _write_wav(path, frames):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"".join(struct.pack("<hh", i % 30000, -(i % 30000)) for i in range(frames)))


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes(), w.readframes(w.getnframes())


def test_master_sample_count_rounds_up():
    # ceil(frames * 1001 / 30000 * 48000)
    for frames in (0, 1, 2, 113, 30000):
        exact = Fraction(frames * 1001 * 48000, 30000)
        expected = math.ceil(exact)
        assert master_sample_count(frames) == expected
        # Last sample ends no earlier than video end, overshoot < one sample.
        got = master_sample_count(frames)
        assert got * 30000 >= frames * 1001 * 48000
        assert (got - 1) * 30000 < frames * 1001 * 48000


def test_loop_is_seamless_and_sample_exact(tmp_path):
    src = tmp_path / "m.wav"
    _write_wav(src, 1000)
    out = prepare_music(src, 4500, tmp_path / "out.wav")
    n, data = _read_wav(out)
    assert n == 4500
    _, src_data = _read_wav(src)
    expected = (src_data * 5)[: 4500 * 4]
    assert data == expected


def test_truncate_when_longer(tmp_path):
    src = tmp_path / "m.wav"
    _write_wav(src, 5000)
    out = prepare_music(src, 1234, tmp_path / "out.wav")
    n, data = _read_wav(out)
    assert n == 1234
    _, src_data = _read_wav(src)
    assert data == src_data[: 1234 * 4]


def test_many_loops_do_not_drift(tmp_path):
    src = tmp_path / "m.wav"
    _write_wav(src, 7)  # awkward length: 30000 loops would expose any drift
    out = prepare_music(src, 7 * 30000 + 3, tmp_path / "out.wav")
    n, data = _read_wav(out)
    assert n == 7 * 30000 + 3
    _, src_data = _read_wav(src)
    assert data == (src_data * 30001)[: len(data)]


def test_rejects_wrong_format(tmp_path):
    bad = tmp_path / "bad.wav"
    with wave.open(str(bad), "wb") as w:
        w.setnchannels(1)  # mono not allowed
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00" * 100)
    with pytest.raises(ValueError):
        prepare_music(bad, 10, tmp_path / "out.wav")
