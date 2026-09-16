"""RMS 音量混合纯函数测试 — CPU torch 即可运行。"""
import unittest

import torch

from rvc.streaming.mix import apply_rms_mix, fast_rms


class TestFastRms(unittest.TestCase):
    def test_silence_near_zero(self):
        wav = torch.zeros(1000)
        rms = fast_rms(wav, frame_length=40, hop_length=10)
        assert torch.all(rms < 1e-3)

    def test_constant_tone(self):
        # 恒定振幅正弦波的 RMS 应接近振幅/sqrt(2)
        t = torch.linspace(0, 1, 16000)
        wav = 0.5 * torch.sin(2 * torch.pi * 440 * t)
        rms = fast_rms(wav, frame_length=160, hop_length=40)
        # 跳过首尾（padding 影响），中间帧应接近 0.35
        middle = rms[5:-5]
        assert torch.all(middle > 0.3)
        assert torch.all(middle < 0.4)

    def test_output_length(self):
        wav = torch.randn(1000)
        rms = fast_rms(wav, frame_length=40, hop_length=10)
        # 反射 padding + avg_pool: (1000 + 40) // 10 ≈ 104
        assert rms.shape[0] > 90
        assert rms.shape[0] < 120


class TestApplyRmsMix(unittest.TestCase):
    def test_rms_mix_one_preserves_converted(self):
        # rms_mix=1.0 表示完全保持转换音频音量
        converted = torch.randn(1000) * 0.5
        reference = torch.randn(1000) * 0.1
        out = apply_rms_mix(reference, converted, rms_mix=1.0, hz_per_centisecond=16)
        torch.testing.assert_close(out, converted, atol=1e-5, rtol=1e-4)

    def test_rms_mix_zero_aligns_to_reference(self):
        # rms_mix=0.0 表示完全跟随参考音量
        converted = torch.ones(1000) * 0.1
        reference = torch.ones(1000) * 0.5
        out = apply_rms_mix(reference, converted, rms_mix=0.0, hz_per_centisecond=16)
        # 输出音量应接近参考（0.5）而非转换（0.1）
        assert out.mean().abs() > 0.3

    def test_output_same_length(self):
        converted = torch.randn(500)
        reference = torch.randn(500)
        out = apply_rms_mix(reference, converted, rms_mix=0.5, hz_per_centisecond=16)
        assert out.shape == converted.shape

    def test_no_nan(self):
        converted = torch.randn(1000)
        reference = torch.randn(1000)
        out = apply_rms_mix(reference, converted, rms_mix=0.5, hz_per_centisecond=16)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))
