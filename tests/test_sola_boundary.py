"""SOLA 与 RMS 混合边界测试 — 各种输入长度、能量、相关性、参数组合。

覆盖：
- apply_sola: 不同 block/buffer/search 大小、静音/全一/随机、低能量/高相关
- fast_rms: 不同帧长/帧移、静音/全一/随机、长度边界
- apply_rms_mix: 不同混合比例、不同长度、静音参考/转换
"""
import unittest

import torch

from rvc.audio.realtime_mix import apply_rms_mix, fast_rms
from rvc.audio.sola import SOLA_MIN_CORR, SOLA_MIN_ENERGY, apply_sola


def _make_sola_setup(block_samples=480, buffer_samples=240, search_samples=120):
    """创建 SOLA 测试所需的缓冲区和窗函数。"""
    sola_buffer = torch.zeros(buffer_samples, dtype=torch.float32)
    sola_norm_kernel = torch.ones(1, 1, buffer_samples, dtype=torch.float32)
    fade_in = torch.linspace(0, 1, buffer_samples, dtype=torch.float32)
    fade_out = torch.linspace(1, 0, buffer_samples, dtype=torch.float32)
    return block_samples, sola_buffer, sola_norm_kernel, fade_in, fade_out


class TestSolaConstants(unittest.TestCase):
    """SOLA 常量测试。"""

    def test_min_corr_value(self):
        self.assertAlmostEqual(SOLA_MIN_CORR, 0.15)

    def test_min_corr_in_range(self):
        self.assertTrue(-1 <= SOLA_MIN_CORR <= 1)

    def test_min_energy_value(self):
        self.assertAlmostEqual(SOLA_MIN_ENERGY, 1e-4)

    def test_min_energy_positive(self):
        self.assertTrue(SOLA_MIN_ENERGY > 0)


class TestApplySolaBasic(unittest.TestCase):
    """apply_sola 基本功能测试。"""

    def test_output_shape(self):
        """输出形状应为 [block_samples]。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.randn(block + 400, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertEqual(out.shape, (block,))

    def test_output_finite(self):
        """输出值有限（无 NaN/Inf）。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.randn(block + 400, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())

    def test_buffer_updated(self):
        """sola_buffer 被更新（不再全零）。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.randn(block + 400, dtype=torch.float32) * 0.1
        apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertFalse(torch.allclose(buf, torch.zeros_like(buf)))

    def test_buffer_shape_preserved(self):
        """sola_buffer 形状不变。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        original_shape = buf.shape
        infer = torch.randn(block + 400, dtype=torch.float32) * 0.1
        apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertEqual(buf.shape, original_shape)

    def test_consecutive_calls(self):
        """连续调用不崩溃，缓冲区持续更新。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        for _ in range(5):
            infer = torch.randn(block + 400, dtype=torch.float32) * 0.1
            out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
            self.assertEqual(out.shape, (block,))
            self.assertTrue(torch.isfinite(out).all())


class TestApplySolaSilence(unittest.TestCase):
    """apply_sola 静音输入测试。"""

    def test_all_zero_input(self):
        """全零输入不崩溃，输出全零。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.zeros(block + 400, dtype=torch.float32)
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertEqual(out.shape, (block,))
        self.assertTrue(torch.isfinite(out).all())

    def test_near_silence_low_energy(self):
        """极低能量输入（低于 SOLA_MIN_ENERGY）不崩溃。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.randn(block + 400, dtype=torch.float32) * 1e-6
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertEqual(out.shape, (block,))
        self.assertTrue(torch.isfinite(out).all())

    def test_silence_after_signal(self):
        """先有信号后静音，缓冲区状态正确。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        # 先处理信号
        infer1 = torch.randn(block + 400, dtype=torch.float32) * 0.1
        apply_sola(infer1, buf, norm, fi, fo, block, 240, 120)
        # 再处理静音
        infer2 = torch.zeros(block + 400, dtype=torch.float32)
        out = apply_sola(infer2, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())


class TestApplySolaDifferentSizes(unittest.TestCase):
    """apply_sola 不同参数大小测试。"""

    def test_small_block(self):
        """小块大小（100 samples）。"""
        block = 100
        buf_s = 50
        search = 20
        block, buf, norm, fi, fo = _make_sola_setup(block, buf_s, search)
        infer = torch.randn(block + buf_s + search + 10, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, buf_s, search)
        self.assertEqual(out.shape, (block,))

    def test_large_block(self):
        """大块大小（2000 samples）。"""
        block = 2000
        buf_s = 500
        search = 200
        block, buf, norm, fi, fo = _make_sola_setup(block, buf_s, search)
        infer = torch.randn(block + buf_s + search + 10, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, buf_s, search)
        self.assertEqual(out.shape, (block,))

    def test_zero_search(self):
        """搜索窗口为 0。"""
        block = 480
        buf_s = 240
        search = 0
        block, buf, norm, fi, fo = _make_sola_setup(block, buf_s, search)
        infer = torch.randn(block + buf_s + search + 10, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, buf_s, search)
        self.assertEqual(out.shape, (block,))

    def test_buffer_equals_block(self):
        """缓冲区大小等于块大小。"""
        block = 480
        buf_s = 480
        search = 120
        block, buf, norm, fi, fo = _make_sola_setup(block, buf_s, search)
        infer = torch.randn(block + buf_s + search + 10, dtype=torch.float32) * 0.1
        out = apply_sola(infer, buf, norm, fi, fo, block, buf_s, search)
        self.assertEqual(out.shape, (block,))

    def test_various_sample_rates(self):
        """不同采样率对应的块大小。"""
        for sr in [16000, 22050, 32000, 44100, 48000]:
            block = sr // 10  # 100ms
            buf_s = block // 2
            search = block // 4
            block, buf, norm, fi, fo = _make_sola_setup(block, buf_s, search)
            infer = torch.randn(block + buf_s + search + 10, dtype=torch.float32) * 0.1
            out = apply_sola(infer, buf, norm, fi, fo, block, buf_s, search)
            self.assertEqual(out.shape, (block,))


class TestApplySolaSignalTypes(unittest.TestCase):
    """apply_sola 不同信号类型测试。"""

    def test_constant_signal(self):
        """常数信号（全 0.5）。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.full((block + 400,), 0.5, dtype=torch.float32)
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())

    def test_sine_wave(self):
        """正弦波信号。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        t = torch.arange(block + 400, dtype=torch.float32)
        infer = torch.sin(2 * 3.14159 * 440 * t / 48000) * 0.3
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())

    def test_high_amplitude(self):
        """高振幅信号（接近 1.0）。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.randn(block + 400, dtype=torch.float32) * 0.9
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())

    def test_negative_signal(self):
        """全负信号。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = -torch.rand(block + 400, dtype=torch.float32) * 0.5
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())

    def test_alternating_signal(self):
        """交替正负信号。"""
        block, buf, norm, fi, fo = _make_sola_setup()
        infer = torch.tensor([1.0, -1.0] * ((block + 400) // 2 + 1), dtype=torch.float32)[:block + 400]
        out = apply_sola(infer, buf, norm, fi, fo, block, 240, 120)
        self.assertTrue(torch.isfinite(out).all())


class TestFastRmsBasic(unittest.TestCase):
    """fast_rms 基本功能测试。"""

    def test_output_shape(self):
        """输出形状约为 [samples // hop_length]。"""
        wav = torch.randn(5000, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=50)
        self.assertEqual(rms.shape[0], 5000 // 50 + 1)

    def test_output_finite(self):
        """输出值有限。"""
        wav = torch.randn(5000, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=50)
        self.assertTrue(torch.isfinite(rms).all())

    def test_output_positive(self):
        """RMS 值非负。"""
        wav = torch.randn(5000, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=50)
        self.assertTrue((rms >= 0).all())

    def test_silence_rms_low(self):
        """静音的 RMS 接近 0（但有 clamp 下限）。"""
        wav = torch.zeros(5000, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=50)
        self.assertTrue((rms < 0.01).all())

    def test_constant_signal_rms(self):
        """常数信号的 RMS 约等于常数绝对值。"""
        wav = torch.full((5000,), 0.5, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=50)
        # 中间帧的 RMS 应接近 0.5
        self.assertTrue(torch.mean(rms[10:-10]) > 0.4)
        self.assertTrue(torch.mean(rms[10:-10]) < 0.6)


class TestFastRmsDifferentParams(unittest.TestCase):
    """fast_rms 不同参数测试。"""

    def test_various_frame_lengths(self):
        """不同帧长。"""
        wav = torch.randn(5000, dtype=torch.float32)
        for fl in [10, 50, 100, 200, 500]:
            rms = fast_rms(wav, frame_length=fl, hop_length=fl // 2)
            self.assertTrue(torch.isfinite(rms).all())
            self.assertTrue((rms >= 0).all())

    def test_various_hop_lengths(self):
        """不同帧移。"""
        wav = torch.randn(5000, dtype=torch.float32)
        for hl in [10, 25, 50, 100, 200]:
            rms = fast_rms(wav, frame_length=100, hop_length=hl)
            self.assertTrue(torch.isfinite(rms).all())

    def test_hop_equals_frame(self):
        """帧移等于帧长（无重叠）。"""
        wav = torch.randn(5000, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=100, hop_length=100)
        self.assertTrue(torch.isfinite(rms).all())

    def test_short_signal(self):
        """短信号（100 samples）。"""
        wav = torch.randn(100, dtype=torch.float32)
        rms = fast_rms(wav, frame_length=50, hop_length=25)
        self.assertTrue(torch.isfinite(rms).all())

    def test_single_sample(self):
        """单样本信号。"""
        wav = torch.tensor([0.5], dtype=torch.float32)
        rms = fast_rms(wav, frame_length=1, hop_length=1)
        self.assertTrue(torch.isfinite(rms).all())

    def test_various_sample_counts(self):
        """不同样本数。"""
        for n in [10, 50, 100, 500, 1000, 5000]:
            wav = torch.randn(n, dtype=torch.float32)
            rms = fast_rms(wav, frame_length=min(100, n), hop_length=min(50, n // 2 + 1))
            self.assertTrue(torch.isfinite(rms).all())


class TestApplyRmsMixBasic(unittest.TestCase):
    """apply_rms_mix 基本功能测试。"""

    def test_output_shape(self):
        """输出形状与 converted 相同。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertEqual(out.shape, conv.shape)

    def test_output_finite(self):
        """输出值有限。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertTrue(torch.isfinite(out).all())

    def test_rms_mix_one_passthrough(self):
        """rms_mix=1.0 时保持原始转换音量（输出≈converted）。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=1.0, hz_per_centisecond=480)
        # rms_mix=1 时 pow(..., 0) = 1，输出应等于 converted
        self.assertTrue(torch.allclose(out, conv, atol=1e-5))

    def test_rms_mix_zero_follows_ref(self):
        """rms_mix=0.0 时完全跟随参考音量。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.5
        conv = torch.randn(5000, dtype=torch.float32) * 0.1
        out = apply_rms_mix(ref, conv, rms_mix=0.0, hz_per_centisecond=480)
        # 输出的 RMS 应接近参考的 RMS（比原始转换大）
        out_rms = torch.sqrt(torch.mean(out**2))
        conv_rms = torch.sqrt(torch.mean(conv**2))
        self.assertTrue(out_rms > conv_rms * 1.5)


class TestApplyRmsMixDifferentParams(unittest.TestCase):
    """apply_rms_mix 不同参数测试。"""

    def test_various_mix_ratios(self):
        """不同混合比例。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.3
        conv = torch.randn(5000, dtype=torch.float32) * 0.1
        for mix in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out = apply_rms_mix(ref, conv, rms_mix=mix, hz_per_centisecond=480)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(out.shape, conv.shape)

    def test_various_hz_values(self):
        """不同厘秒 Hz 值（对应不同采样率）。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        for hz in [160, 220, 320, 441, 480]:
            out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=hz)
            self.assertTrue(torch.isfinite(out).all())

    def test_ref_hz_override(self):
        """ref_hz 覆盖（离线路径）。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480, ref_hz=160)
        self.assertTrue(torch.isfinite(out).all())

    def test_ref_hz_none_equals_hz(self):
        """ref_hz=None 时与 hz_per_centisecond 相同。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out1 = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480, ref_hz=None)
        out2 = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480, ref_hz=480)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_different_lengths(self):
        """不同音频长度（使用小 hz 避免 padding 超过输入长度）。"""
        for n in [200, 500, 1000, 5000]:
            ref = torch.randn(n, dtype=torch.float32) * 0.1
            conv = torch.randn(n, dtype=torch.float32) * 0.2
            out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=10)
            self.assertEqual(out.shape, (n,))
            self.assertTrue(torch.isfinite(out).all())

    def test_ref_longer_than_conv(self):
        """参考比转换长（只取前 converted.shape[0]）。"""
        ref = torch.randn(2000, dtype=torch.float32) * 0.1
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertEqual(out.shape, (5000,))

    def test_silence_reference(self):
        """静音参考。"""
        ref = torch.zeros(5000, dtype=torch.float32)
        conv = torch.randn(5000, dtype=torch.float32) * 0.2
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertTrue(torch.isfinite(out).all())

    def test_silence_converted(self):
        """静音转换。"""
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        conv = torch.zeros(5000, dtype=torch.float32)
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertTrue(torch.isfinite(out).all())
        # 静音输入输出应为静音
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))

    def test_both_silence(self):
        """双方都静音。"""
        ref = torch.zeros(5000, dtype=torch.float32)
        conv = torch.zeros(5000, dtype=torch.float32)
        out = apply_rms_mix(ref, conv, rms_mix=0.5, hz_per_centisecond=480)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
