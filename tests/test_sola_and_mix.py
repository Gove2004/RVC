"""SOLA 对齐与 RMS 混合单元测试 — 纯函数，边界条件。

全部用 CPU 设备，不依赖 GPU。测试数据用确定性张量，验证形状、数值范围、
边界条件和状态更新正确性。
"""
import unittest

import torch

from rvc.audio.sola import apply_sola, SOLA_MIN_CORR, SOLA_MIN_ENERGY
from rvc.audio.realtime_mix import fast_rms, apply_rms_mix


class TestFastRMS(unittest.TestCase):
    """fast_rms 数值正确性测试。"""

    def test_output_shape(self):
        """输出帧数 = floor((input_len + 2*padding - kernel) / stride) + 1。"""
        wav = torch.randn(16000)
        rms = fast_rms(wav, frame_length=640, hop_length=160)
        # input after padding = 16000 + 320*2 = 16640
        expected = (16640 - 640) // 160 + 1
        self.assertEqual(rms.shape[0], expected)

    def test_silence_positive(self):
        """静音输入 RMS > 0（clamp 到 1e-8，sqrt 后约 1e-4）。"""
        wav = torch.zeros(16000)
        rms = fast_rms(wav, frame_length=640, hop_length=160)
        self.assertTrue(torch.all(rms > 0))
        self.assertTrue(torch.all(rms < 1e-3))  # sqrt(1e-8) ≈ 1e-4

    def test_sine_wave_rms(self):
        """正弦波 RMS 应接近 1/sqrt(2) ≈ 0.707（幅值=1）。"""
        sr = 16000
        t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
        sine = torch.sin(2 * torch.pi * 440 * t)
        rms = fast_rms(sine, frame_length=640, hop_length=160)
        mid = rms[10:-10]
        self.assertTrue(torch.allclose(mid, torch.tensor(0.707), atol=0.05))

    def test_dc_offset(self):
        """直流偏移信号 RMS 等于偏移量（全 1 的 RMS = 1）。"""
        wav = torch.ones(16000)
        rms = fast_rms(wav, frame_length=640, hop_length=160)
        mid = rms[10:-10]
        self.assertTrue(torch.allclose(mid, torch.tensor(1.0), atol=0.01))

    def test_short_input(self):
        """输入长度接近 frame_length 时不崩溃。"""
        wav = torch.randn(700)
        rms = fast_rms(wav, frame_length=640, hop_length=160)
        self.assertGreaterEqual(rms.shape[0], 1)


class TestApplyRmsMix(unittest.TestCase):
    """apply_rms_mix 混合比例与边界测试。"""

    def setUp(self):
        self.sr = 16000
        self.hz = self.sr // 100  # 160
        t = torch.linspace(0, 1.0, self.sr, dtype=torch.float32)
        self.reference = torch.sin(2 * torch.pi * 440 * t) * 0.5
        self.converted = torch.sin(2 * torch.pi * 440 * t) * 0.1

    def test_output_shape(self):
        """输出形状与 converted 相同。"""
        out = apply_rms_mix(self.reference, self.converted, 0.5, self.hz)
        self.assertEqual(out.shape, self.converted.shape)

    def test_rms_mix_one_no_change(self):
        """rms_mix=1 时输出等于 converted（指数=0，pow=1）。"""
        out = apply_rms_mix(self.reference, self.converted, 1.0, self.hz)
        self.assertTrue(torch.allclose(out, self.converted, atol=1e-5))

    def test_rms_mix_zero_adjusts(self):
        """rms_mix=0 时输出音量被调整（不等于 converted）。"""
        out = apply_rms_mix(self.reference, self.converted, 0.0, self.hz)
        # reference 音量 (0.5) > converted 音量 (0.1)，调整后输出应更大
        out_rms = fast_rms(out, 4 * self.hz, self.hz).mean()
        conv_rms = fast_rms(self.converted, 4 * self.hz, self.hz).mean()
        self.assertGreater(out_rms.item(), conv_rms.item() * 1.5)

    def test_silence_converted_no_crash(self):
        """converted 为静音时不崩溃（r2 被 clamp 到 1e-3）。"""
        silent = torch.zeros_like(self.converted)
        out = apply_rms_mix(self.reference, silent, 0.5, self.hz)
        self.assertEqual(out.shape, silent.shape)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_reference_longer(self):
        """reference 比 converted 长时不崩溃（截断到 converted 长度）。"""
        long_ref = torch.cat([self.reference, self.reference])
        out = apply_rms_mix(long_ref, self.converted, 0.5, self.hz)
        self.assertEqual(out.shape, self.converted.shape)

    def test_reference_shorter(self):
        """reference 比 converted 短时不崩溃。"""
        short_ref = self.reference[:1000]
        out = apply_rms_mix(short_ref, self.converted, 0.5, self.hz)
        self.assertEqual(out.shape, self.converted.shape)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_output_finite(self):
        """输出全部为有限值（无 NaN/Inf）。"""
        out = apply_rms_mix(self.reference, self.converted, 0.5, self.hz)
        self.assertTrue(torch.all(torch.isfinite(out)))


class TestApplySola(unittest.TestCase):
    """apply_sola 形状与边界测试。"""

    def setUp(self):
        self.block_samples = 256
        self.sola_buffer_samples = 64
        self.sola_search_samples = 32
        # 输入长度必须 >= block_samples + sola_search_samples + sola_buffer_samples
        # （offset 后需要覆盖完整输出块 + 下一块的 sola_buffer）
        self.infer_length = self.block_samples + self.sola_search_samples + self.sola_buffer_samples + 10

    def _make_inputs(self):
        infer = torch.randn(self.infer_length)
        sola_buffer = torch.randn(self.sola_buffer_samples)
        # conv1d 权重需要 3 维 [out_ch, in_ch/groups, kernel_size]
        sola_norm_kernel = torch.ones(1, 1, self.sola_buffer_samples)
        fade_in = torch.linspace(0, 1, self.sola_buffer_samples)
        fade_out = torch.linspace(1, 0, self.sola_buffer_samples)
        return infer, sola_buffer, sola_norm_kernel, fade_in, fade_out

    def test_output_shape(self):
        """输出形状 = [block_samples]。"""
        infer, buf, kernel, fin, fout = self._make_inputs()
        out = apply_sola(
            infer, buf, kernel, fin, fout,
            self.block_samples, self.sola_buffer_samples, self.sola_search_samples,
        )
        self.assertEqual(out.shape[0], self.block_samples)

    def test_sola_buffer_updated(self):
        """sola_buffer 被原地修改。"""
        infer, buf, kernel, fin, fout = self._make_inputs()
        original = buf.clone()
        apply_sola(
            infer, buf, kernel, fin, fout,
            self.block_samples, self.sola_buffer_samples, self.sola_search_samples,
        )
        self.assertFalse(torch.allclose(buf, original))
        self.assertEqual(buf.shape[0], self.sola_buffer_samples)

    def test_silence_no_crash(self):
        """静音输入不崩溃（energy < 阈值时 offset=0）。"""
        infer = torch.zeros(self.infer_length)
        buf = torch.zeros(self.sola_buffer_samples)
        kernel = torch.ones(1, 1, self.sola_buffer_samples)
        fin = torch.linspace(0, 1, self.sola_buffer_samples)
        fout = torch.linspace(1, 0, self.sola_buffer_samples)
        out = apply_sola(
            infer, buf, kernel, fin, fout,
            self.block_samples, self.sola_buffer_samples, self.sola_search_samples,
        )
        self.assertEqual(out.shape[0], self.block_samples)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_output_finite(self):
        """输出全部为有限值。"""
        infer, buf, kernel, fin, fout = self._make_inputs()
        out = apply_sola(
            infer, buf, kernel, fin, fout,
            self.block_samples, self.sola_buffer_samples, self.sola_search_samples,
        )
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_constants(self):
        """SOLA 阈值常量存在且值合理。"""
        self.assertGreater(SOLA_MIN_CORR, 0)
        self.assertLess(SOLA_MIN_CORR, 1)
        self.assertGreater(SOLA_MIN_ENERGY, 0)


if __name__ == "__main__":
    unittest.main()
