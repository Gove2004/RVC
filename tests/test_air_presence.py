"""空气感效果器单元测试 — High Shelf 高频搁架提升。"""
import math
import unittest

import numpy as np
import torch

from rvc.audio.effects import AirPresenceEffect, AudioProcessor


class TestAirPresenceEffect(unittest.TestCase):
    """AirPresenceEffect 单元测试。"""

    def setUp(self):
        self.sr = 48000
        self.block_samples = 4800  # 100ms
        self.device = "cpu"
        self.effect = AirPresenceEffect()
        self.effect.setup(self.sr, self.block_samples, self.device)

    def _stabilize(self, audio, strength, n_blocks=5):
        """处理多个块让滤波器和参数平滑稳定。"""
        for _ in range(n_blocks):
            self.effect.process(audio, strength)

    def test_strength_zero_returns_input(self):
        """strength=0 时输出等于输入（直通）。"""
        audio = torch.randn(self.block_samples)
        out = self.effect.process(audio, 0.0)
        self.assertTrue(torch.allclose(out, audio, atol=1e-6))

    def test_strength_positive_changes_output(self):
        """strength>0 时输出与输入不同。"""
        audio = torch.randn(self.block_samples)
        out = self.effect.process(audio, 50.0)
        self.assertFalse(torch.allclose(out, audio, atol=1e-3))

    def test_high_frequency_boosted(self):
        """高频信号被提升（用白噪声+频谱分析验证）。"""
        # 生成白噪声
        torch.manual_seed(42)
        audio = torch.randn(self.block_samples * 4) * 0.3

        # 稳定滤波器
        self._stabilize(audio[:self.block_samples], 100.0, n_blocks=3)

        # 处理一个块
        out = self.effect.process(audio[self.block_samples:2*self.block_samples], 100.0)

        # 计算高频段（8kHz 以上）能量比
        def high_freq_energy(signal, sr, cutoff=8000):
            # 简单的 FFT 分析
            n = len(signal)
            freqs = torch.fft.rfftfreq(n, d=1.0/sr)
            spectrum = torch.abs(torch.fft.rfft(signal))
            high_mask = freqs >= cutoff
            return torch.sum(spectrum[high_mask] ** 2).item()

        in_high = high_freq_energy(audio[self.block_samples:2*self.block_samples], self.sr)
        out_high = high_freq_energy(out, self.sr)

        # 输出高频能量应该大于输入
        self.assertGreater(out_high, in_high * 1.2)  # 至少提升 20%

    def test_low_frequency_unchanged(self):
        """低频信号基本不变（1kHz 以下增益 ≈ 0dB）。"""
        # 生成 500Hz 正弦波（低频）
        t = torch.arange(self.block_samples * 2) / self.sr
        low_freq = torch.sin(2 * math.pi * 500 * t) * 0.5

        # 稳定滤波器
        self._stabilize(low_freq[:self.block_samples], 100.0, n_blocks=5)
        out = self.effect.process(low_freq[self.block_samples:], 100.0)

        # 输出幅度应该接近输入（低频不提升）
        in_rms = torch.sqrt(torch.mean(low_freq[self.block_samples+1000:] ** 2))
        out_rms = torch.sqrt(torch.mean(out[1000:] ** 2))
        self.assertLess(abs(out_rms - in_rms), in_rms * 0.1)  # 变化 < 10%

    def test_causal_filter(self):
        """滤波器是因果的：输出[n] 不依赖输入[n+1]。"""
        torch.manual_seed(42)
        audio1 = torch.randn(self.block_samples)
        audio2 = audio1.clone()
        # 修改后半部分
        audio2[self.block_samples // 2:] = torch.randn(self.block_samples // 2)

        # 重置状态，确保两者初始状态相同
        self.effect.reset()
        out1 = self.effect.process(audio1, 50.0)

        self.effect.reset()
        out2 = self.effect.process(audio2, 50.0)

        # 前半部分输出应该相同（因为输入前半部分相同，且初始状态相同）
        self.assertTrue(torch.allclose(
            out1[:self.block_samples // 2],
            out2[:self.block_samples // 2],
            atol=1e-5
        ))

    def test_reset_clears_state(self):
        """reset() 后延迟线清零，输出不依赖历史。"""
        audio = torch.ones(self.block_samples) * 0.5
        self.effect.process(audio, 50.0)  # 填充延迟线

        self.effect.reset()
        # 重置后处理静音，输出应该快速衰减到 0
        silence = torch.zeros(self.block_samples)
        out = self.effect.process(silence, 50.0)
        self.assertLess(torch.max(torch.abs(out[500:])), 0.01)

    def test_output_shape(self):
        """输出形状与输入相同。"""
        for n in [100, 480, 4800, 9600]:
            audio = torch.randn(n)
            out = self.effect.process(audio, 50.0)
            self.assertEqual(out.shape, audio.shape)

    def test_no_nan_no_inf(self):
        """输出不包含 NaN 或 Inf。"""
        audio = torch.randn(self.block_samples)
        out = self.effect.process(audio, 100.0)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_smoothing_no_pop(self):
        """参数变化时平滑过渡，不会产生爆音（幅度突变 < 10%）。"""
        # 先稳定在 strength=0
        audio = torch.ones(self.block_samples) * 0.3
        self._stabilize(audio, 0.0, n_blocks=3)

        # 突然切换到 strength=100
        out = self.effect.process(audio, 100.0)

        # 输出应该平滑变化，不会有突变
        diff = torch.abs(torch.diff(out))
        self.assertLess(torch.max(diff), 0.1)  # 相邻样本差 < 0.1

    def test_different_sample_rates(self):
        """不同采样率下都能正常工作。"""
        for sr in [16000, 22050, 44100, 48000]:
            effect = AirPresenceEffect()
            effect.setup(sr, sr // 10, "cpu")
            audio = torch.randn(sr // 10)
            out = effect.process(audio, 50.0)
            self.assertEqual(out.shape, audio.shape)
            self.assertFalse(torch.isnan(out).any())

    def test_max_gain_12db(self):
        """strength=100 对应最大 +12dB 增益。"""
        self.assertAlmostEqual(self.effect.MAX_GAIN_DB, 12.0, places=1)

    def test_cutoff_8khz(self):
        """截止频率为 8kHz。"""
        self.assertAlmostEqual(self.effect.CUTOFF_HZ, 8000.0, places=1)

    def test_reset_resets_gain(self):
        """reset() 后当前增益重置为 0。"""
        self._stabilize(torch.ones(self.block_samples) * 0.3, 100.0, n_blocks=10)
        self.assertGreater(self.effect._current_gain_db, 1.0)  # 已经有增益
        self.effect.reset()
        self.assertEqual(self.effect._current_gain_db, 0.0)


class TestAudioProcessorAirPresence(unittest.TestCase):
    """AudioProcessor 中空气感集成测试。"""

    def setUp(self):
        self.sr = 48000
        self.block_samples = 4800
        self.crossfade_samples = int(0.05 * self.sr)
        self.sola_buffer_samples = min(self.crossfade_samples, 4 * (self.sr // 100))
        self.sola_search_samples = self.sr // 100
        # SOLA 需要的输入长度：block + sola_buffer + sola_search
        self.input_length = self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        self.device = "cpu"
        self.processor = AudioProcessor()
        self.processor.setup(
            self.sr, self.block_samples,
            crossfade_samples=self.crossfade_samples,
            sola_search_samples=self.sola_search_samples,
            device=self.device,
        )

    def test_process_output_with_air_presence(self):
        """process_output 支持 air_presence 参数。"""
        infer = torch.randn(self.input_length)
        ref = torch.randn(self.input_length)
        out = self.processor.process_output(
            infer, ref, rms_mix=1.0, is_vc=True, air_presence=50.0,
        )
        self.assertIsNotNone(out)
        self.assertFalse(torch.isnan(out).any())

    def test_air_presence_zero_same_as_without(self):
        """air_presence=0 时与不传该参数效果相同（SOLA 后）。"""
        torch.manual_seed(42)
        infer = torch.randn(self.input_length)
        ref = torch.randn(self.input_length)

        self.processor.reset()
        out1 = self.processor.process_output(infer, ref, rms_mix=1.0, is_vc=True)

        self.processor.reset()
        out2 = self.processor.process_output(infer, ref, rms_mix=1.0, is_vc=True, air_presence=0.0)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_reset_clears_air_presence_state(self):
        """AudioProcessor.reset() 会重置空气感状态。"""
        audio = torch.ones(self.input_length) * 0.5
        self.processor.process_output(audio, audio, rms_mix=1.0, is_vc=True, air_presence=50.0)
        self.processor.reset()
        # 重置后空气感延迟线应该清零
        self.assertEqual(self.processor.air_presence._x1, 0.0)
        self.assertEqual(self.processor.air_presence._y1, 0.0)
        self.assertEqual(self.processor.air_presence._current_gain_db, 0.0)


if __name__ == "__main__":
    unittest.main()
