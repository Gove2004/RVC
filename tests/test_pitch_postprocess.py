"""音高后处理纯函数测试 — 不依赖 torch CUDA，CPU 即可运行。"""
import unittest

import numpy as np
import torch

from rvc.pipeline.pitch.postprocess import (
    RMVPE_THRESHOLD,
    apply_pitch_map,
    hz_to_midi,
    median_filter_f0,
    midi_to_hz,
    normalize_f0_to_coarse,
)


class TestHzMidiConversion(unittest.TestCase):
    def test_a4_reference(self):
        assert abs(hz_to_midi(440.0) - 69.0) < 1e-6

    def test_roundtrip(self):
        for freq in [110, 220, 440, 880, 1046.5]:
            midi = hz_to_midi(freq)
            back = midi_to_hz(midi)
            assert abs(back - freq) < 1e-4

    def test_octave_relation(self):
        # 880Hz 比 440Hz 高 12 个半音
        assert abs(hz_to_midi(880.0) - hz_to_midi(440.0) - 12.0) < 1e-6


class TestApplyPitchMap(unittest.TestCase):
    def test_identity_mapping(self):
        f0 = np.array([220.0, 440.0, 880.0], dtype=np.float32)
        out = apply_pitch_map(f0, 100, 1000, 100, 1000)
        np.testing.assert_allclose(out, f0, rtol=1e-5)

    def test_octave_shift_up(self):
        # 源 100-1000 → 目标 200-2000 = 整体上移一个八度
        f0 = np.array([440.0], dtype=np.float32)
        out = apply_pitch_map(f0, 100, 1000, 200, 2000)
        assert abs(out[0] - 880.0) < 1.0

    def test_silent_frames_preserved(self):
        f0 = np.array([0.0, 440.0, 0.0], dtype=np.float32)
        out = apply_pitch_map(f0, 100, 1000, 200, 2000)
        assert out[0] == 0.0
        assert out[2] == 0.0

    def test_invalid_params_return_unchanged(self):
        f0 = np.array([440.0], dtype=np.float32)
        # src_min >= src_max
        out = apply_pitch_map(f0, 500, 100, 200, 2000)
        np.testing.assert_array_equal(out, f0)

    def test_zero_freq_clamped(self):
        # 参数为 0 时应被钳制到 1Hz，不产生 NaN
        f0 = np.array([440.0], dtype=np.float32)
        out = apply_pitch_map(f0, 0, 1000, 0, 2000)
        assert not np.any(np.isnan(out))

    def test_torch_input(self):
        f0 = torch.tensor([440.0, 880.0])
        out = apply_pitch_map(f0, 100, 1000, 100, 1000)
        assert torch.is_tensor(out)
        torch.testing.assert_close(out, f0, rtol=1e-4, atol=1e-4)


class TestMedianFilterF0(unittest.TestCase):
    def test_removes_spike(self):
        f0 = torch.tensor([100.0, 100.0, 500.0, 100.0, 100.0])
        out = median_filter_f0(f0, kernel=3)
        # 中间的 500 尖峰应被中值滤波压平
        assert out[2] == 100.0

    def test_preserves_constant(self):
        f0 = torch.full((10,), 440.0)
        out = median_filter_f0(f0, kernel=3)
        torch.testing.assert_close(out, f0)

    def test_silent_frames(self):
        f0 = torch.tensor([0.0, 0.0, 440.0, 0.0, 0.0])
        out = median_filter_f0(f0, kernel=3)
        # 0 值（清音）不应被滤波改变
        assert out[0] == 0.0
        assert out[4] == 0.0


class TestNormalizeF0ToCoarse(unittest.TestCase):
    def test_basic_shape(self):
        f0 = np.array([0.0, 220.0, 440.0, 880.0], dtype=np.float32)
        coarse = normalize_f0_to_coarse(f0)
        assert coarse.shape == f0.shape
        # 静音帧映射为 1（不是 0）
        assert coarse[0] == 1

    def test_monotonic_with_freq(self):
        f0 = np.array([220.0, 440.0, 880.0], dtype=np.float32)
        coarse = normalize_f0_to_coarse(f0)
        assert coarse[0] < coarse[1] < coarse[2]


class TestConstants(unittest.TestCase):
    def test_rmvpe_threshold(self):
        assert RMVPE_THRESHOLD == 0.03
