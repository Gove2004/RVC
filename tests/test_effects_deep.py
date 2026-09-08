"""效果器深度测试 — RmsMixEffect / SolaEffect / AudioProcessor 各种边界条件。

覆盖：
- RmsMixEffect: setup/process、各种 rms_mix 值、各种采样率、直通模式
- SolaEffect: setup/process/reset、各种 block/crossfade/search 大小、连续处理
- AudioProcessor: setup/process_output/reset、is_vc 开关、各种参数组合
"""
import unittest

import torch

from rvc.audio.effects import AudioProcessor, RmsMixEffect, SolaEffect


class TestRmsMixEffectInit(unittest.TestCase):
    """RmsMixEffect 初始化测试。"""

    def test_default_hz_zero(self):
        effect = RmsMixEffect()
        self.assertEqual(effect._hz_centis, 0)

    def test_has_setup_method(self):
        effect = RmsMixEffect()
        self.assertTrue(hasattr(effect, "setup"))
        self.assertTrue(callable(effect.setup))

    def test_has_process_method(self):
        effect = RmsMixEffect()
        self.assertTrue(hasattr(effect, "process"))
        self.assertTrue(callable(effect.process))


class TestRmsMixEffectSetup(unittest.TestCase):
    """RmsMixEffect setup 测试。"""

    def test_setup_16k(self):
        effect = RmsMixEffect()
        effect.setup(sr=16000)
        self.assertEqual(effect._hz_centis, 160)

    def test_setup_22k(self):
        effect = RmsMixEffect()
        effect.setup(sr=22050)
        self.assertEqual(effect._hz_centis, 220)

    def test_setup_32k(self):
        effect = RmsMixEffect()
        effect.setup(sr=32000)
        self.assertEqual(effect._hz_centis, 320)

    def test_setup_44k(self):
        effect = RmsMixEffect()
        effect.setup(sr=44100)
        self.assertEqual(effect._hz_centis, 441)

    def test_setup_48k(self):
        effect = RmsMixEffect()
        effect.setup(sr=48000)
        self.assertEqual(effect._hz_centis, 480)

    def test_setup_8k(self):
        effect = RmsMixEffect()
        effect.setup(sr=8000)
        self.assertEqual(effect._hz_centis, 80)

    def test_setup_96k(self):
        effect = RmsMixEffect()
        effect.setup(sr=96000)
        self.assertEqual(effect._hz_centis, 960)

    def test_multiple_setups(self):
        effect = RmsMixEffect()
        effect.setup(sr=16000)
        self.assertEqual(effect._hz_centis, 160)
        effect.setup(sr=48000)
        self.assertEqual(effect._hz_centis, 480)


class TestRmsMixEffectProcess(unittest.TestCase):
    """RmsMixEffect process 测试。"""

    def setUp(self):
        self.effect = RmsMixEffect()
        self.effect.setup(sr=16000)

    def test_rms_mix_one_passthrough(self):
        """rms_mix >= 1.0 时直通，返回原 tensor。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.2
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        out = self.effect.process(infer, ref, rms_mix=1.0)
        self.assertIs(out, infer)

    def test_rms_mix_above_one_passthrough(self):
        """rms_mix > 1.0 时也直通。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.2
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        out = self.effect.process(infer, ref, rms_mix=2.0)
        self.assertIs(out, infer)

    def test_rms_mix_zero(self):
        """rms_mix=0 时完全跟随参考音量。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.05
        ref = torch.randn(5000, dtype=torch.float32) * 0.5
        out = self.effect.process(infer, ref, rms_mix=0.0)
        self.assertEqual(out.shape, infer.shape)
        self.assertTrue(torch.isfinite(out).all())
        # 输出能量应该比原始 infer 大（跟随 ref 的大音量）
        self.assertTrue(torch.mean(out**2) > torch.mean(infer**2))

    def test_rms_mix_half(self):
        """rms_mix=0.5 时混合。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.randn(5000, dtype=torch.float32) * 0.3
        out = self.effect.process(infer, ref, rms_mix=0.5)
        self.assertEqual(out.shape, infer.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_various_rms_mix_values(self):
        """各种 rms_mix 值。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.randn(5000, dtype=torch.float32) * 0.3
        for mix in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            out = self.effect.process(infer.clone(), ref, rms_mix=mix)
            self.assertEqual(out.shape, infer.shape)
            self.assertTrue(torch.isfinite(out).all())

    def test_silence_reference(self):
        """静音参考。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.zeros(5000, dtype=torch.float32)
        out = self.effect.process(infer, ref, rms_mix=0.5)
        self.assertTrue(torch.isfinite(out).all())

    def test_silence_infer(self):
        """静音 infer。"""
        infer = torch.zeros(5000, dtype=torch.float32)
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        out = self.effect.process(infer, ref, rms_mix=0.5)
        self.assertTrue(torch.isfinite(out).all())
        # 静音输入输出应为静音
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))

    def test_both_silence(self):
        """双方都静音。"""
        infer = torch.zeros(5000, dtype=torch.float32)
        ref = torch.zeros(5000, dtype=torch.float32)
        out = self.effect.process(infer, ref, rms_mix=0.5)
        self.assertTrue(torch.isfinite(out).all())

    def test_various_lengths(self):
        """各种输入长度。"""
        for n in [500, 1000, 2000, 5000, 10000]:
            infer = torch.randn(n, dtype=torch.float32) * 0.1
            ref = torch.randn(n, dtype=torch.float32) * 0.2
            out = self.effect.process(infer, ref, rms_mix=0.5)
            self.assertEqual(out.shape, (n,))
            self.assertTrue(torch.isfinite(out).all())

    def test_output_dtype(self):
        """输出 dtype 为 float32。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.randn(5000, dtype=torch.float32) * 0.2
        out = self.effect.process(infer, ref, rms_mix=0.5)
        self.assertEqual(out.dtype, torch.float32)


class TestSolaEffectInit(unittest.TestCase):
    """SolaEffect 初始化测试。"""

    def test_default_buffer_none(self):
        effect = SolaEffect()
        self.assertIsNone(effect.sola_buffer)

    def test_default_fade_none(self):
        effect = SolaEffect()
        self.assertIsNone(effect._fade_in)
        self.assertIsNone(effect._fade_out)

    def test_default_norm_kernel_none(self):
        effect = SolaEffect()
        self.assertIsNone(effect._sola_norm_kernel)

    def test_default_block_zero(self):
        effect = SolaEffect()
        self.assertEqual(effect._block_samples, 0)

    def test_has_setup_method(self):
        effect = SolaEffect()
        self.assertTrue(hasattr(effect, "setup"))
        self.assertTrue(callable(effect.setup))

    def test_has_process_method(self):
        effect = SolaEffect()
        self.assertTrue(hasattr(effect, "process"))
        self.assertTrue(callable(effect.process))

    def test_has_reset_method(self):
        effect = SolaEffect()
        self.assertTrue(hasattr(effect, "reset"))
        self.assertTrue(callable(effect.reset))


class TestSolaEffectSetup(unittest.TestCase):
    """SolaEffect setup 测试。"""

    def test_setup_basic(self):
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertEqual(effect._block_samples, 480)
        self.assertEqual(effect._sola_search_samples, 120)
        self.assertIsNotNone(effect.sola_buffer)
        self.assertIsNotNone(effect._fade_in)
        self.assertIsNotNone(effect._fade_out)
        self.assertIsNotNone(effect._sola_norm_kernel)

    def test_buffer_shape(self):
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        # crossfade=240, 4*zc=4*480=1920, min=240
        self.assertEqual(effect.sola_buffer.shape, (240,))

    def test_buffer_capped_at_4zc(self):
        """crossfade 超过 4*zc 时被截断。"""
        effect = SolaEffect()
        effect.setup(sr=16000, block_samples=160, crossfade_samples=1000,
                     sola_search_samples=80, device="cpu")
        # 4*zc=4*160=640, crossfade=1000, min=640
        self.assertEqual(effect.sola_buffer.shape, (640,))

    def test_fade_in_shape(self):
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertEqual(effect._fade_in.shape, (240,))

    def test_fade_out_shape(self):
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertEqual(effect._fade_out.shape, (240,))

    def test_norm_kernel_shape(self):
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertEqual(effect._sola_norm_kernel.shape, (1, 1, 240))

    def test_fade_in_range(self):
        """fade_in 值在 [0, 1] 范围内。"""
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertTrue((effect._fade_in >= 0).all())
        self.assertTrue((effect._fade_in <= 1).all())

    def test_fade_out_range(self):
        """fade_out 值在 [0, 1] 范围内。"""
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        self.assertTrue((effect._fade_out >= 0).all())
        self.assertTrue((effect._fade_out <= 1).all())

    def test_fade_in_out_sum(self):
        """fade_in + fade_out ≈ 1。"""
        effect = SolaEffect()
        effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                     sola_search_samples=120, device="cpu")
        total = effect._fade_in + effect._fade_out
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    def test_various_sample_rates(self):
        """各种采样率 setup。"""
        for sr in [16000, 22050, 32000, 44100, 48000]:
            effect = SolaEffect()
            block = sr // 100
            effect.setup(sr=sr, block_samples=block, crossfade_samples=block//2,
                         sola_search_samples=block//4, device="cpu")
            self.assertIsNotNone(effect.sola_buffer)
            self.assertEqual(effect._block_samples, block)

    def test_various_block_sizes(self):
        """各种块大小。"""
        for block in [128, 256, 480, 512, 960, 1024]:
            effect = SolaEffect()
            effect.setup(sr=48000, block_samples=block, crossfade_samples=block//2,
                         sola_search_samples=block//4, device="cpu")
            self.assertEqual(effect._block_samples, block)
            self.assertIsNotNone(effect.sola_buffer)


class TestSolaEffectProcess(unittest.TestCase):
    """SolaEffect process 测试。"""

    def setUp(self):
        self.effect = SolaEffect()
        self.effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                          sola_search_samples=120, device="cpu")

    def _make_input(self, n=850):
        return torch.randn(n, dtype=torch.float32) * 0.1

    def test_output_shape(self):
        infer = self._make_input()
        out = self.effect.process(infer)
        self.assertEqual(out.shape, (480,))

    def test_output_finite(self):
        infer = self._make_input()
        out = self.effect.process(infer)
        self.assertTrue(torch.isfinite(out).all())

    def test_output_dtype(self):
        infer = self._make_input()
        out = self.effect.process(infer)
        self.assertEqual(out.dtype, torch.float32)

    def test_buffer_updated(self):
        infer = self._make_input()
        self.effect.process(infer)
        # 处理后 buffer 不应全零
        self.assertFalse(torch.allclose(self.effect.sola_buffer, torch.zeros_like(self.effect.sola_buffer)))

    def test_consecutive_calls(self):
        """连续调用不崩溃。"""
        for _ in range(10):
            infer = self._make_input()
            out = self.effect.process(infer)
            self.assertEqual(out.shape, (480,))
            self.assertTrue(torch.isfinite(out).all())

    def test_long_sequence(self):
        """长序列处理（100 帧）。"""
        for i in range(100):
            infer = self._make_input()
            out = self.effect.process(infer)
            self.assertEqual(out.shape, (480,))
            self.assertTrue(torch.isfinite(out).all())

    def test_silence_input(self):
        infer = torch.zeros(850, dtype=torch.float32)
        out = self.effect.process(infer)
        self.assertTrue(torch.isfinite(out).all())

    def test_constant_input(self):
        infer = torch.full((850,), 0.5, dtype=torch.float32)
        out = self.effect.process(infer)
        self.assertTrue(torch.isfinite(out).all())

    def test_high_amplitude(self):
        infer = torch.randn(850, dtype=torch.float32) * 0.9
        out = self.effect.process(infer)
        self.assertTrue(torch.isfinite(out).all())


class TestSolaEffectReset(unittest.TestCase):
    """SolaEffect reset 测试。"""

    def setUp(self):
        self.effect = SolaEffect()
        self.effect.setup(sr=48000, block_samples=480, crossfade_samples=240,
                          sola_search_samples=120, device="cpu")

    def test_reset_clears_buffer(self):
        # 先处理一帧，让 buffer 非零
        infer = torch.randn(850, dtype=torch.float32) * 0.1
        self.effect.process(infer)
        self.assertFalse(torch.allclose(self.effect.sola_buffer, torch.zeros_like(self.effect.sola_buffer)))
        # reset 后 buffer 应为零
        self.effect.reset()
        self.assertTrue(torch.allclose(self.effect.sola_buffer, torch.zeros_like(self.effect.sola_buffer)))

    def test_reset_before_setup_no_crash(self):
        """未 setup 时 reset 不崩溃。"""
        effect = SolaEffect()
        effect.reset()  # 不应崩溃

    def test_reset_then_process(self):
        """reset 后继续处理正常。"""
        infer = torch.randn(850, dtype=torch.float32) * 0.1
        self.effect.process(infer)
        self.effect.reset()
        out = self.effect.process(infer)
        self.assertEqual(out.shape, (480,))
        self.assertTrue(torch.isfinite(out).all())

    def test_multiple_resets(self):
        """多次 reset 不崩溃。"""
        for _ in range(5):
            infer = torch.randn(850, dtype=torch.float32) * 0.1
            self.effect.process(infer)
            self.effect.reset()
            self.assertTrue(torch.allclose(self.effect.sola_buffer, torch.zeros_like(self.effect.sola_buffer)))


class TestAudioProcessorInit(unittest.TestCase):
    """AudioProcessor 初始化测试。"""

    def test_has_rms_mix(self):
        proc = AudioProcessor()
        self.assertIsInstance(proc.rms_mix, RmsMixEffect)

    def test_has_sola(self):
        proc = AudioProcessor()
        self.assertIsInstance(proc.sola, SolaEffect)

    def test_has_setup_method(self):
        proc = AudioProcessor()
        self.assertTrue(hasattr(proc, "setup"))
        self.assertTrue(callable(proc.setup))

    def test_has_process_output_method(self):
        proc = AudioProcessor()
        self.assertTrue(hasattr(proc, "process_output"))
        self.assertTrue(callable(proc.process_output))

    def test_has_reset_method(self):
        proc = AudioProcessor()
        self.assertTrue(hasattr(proc, "reset"))
        self.assertTrue(callable(proc.reset))


class TestAudioProcessorSetup(unittest.TestCase):
    """AudioProcessor setup 测试。"""

    def test_setup_basic(self):
        proc = AudioProcessor()
        proc.setup(sr=48000, block_samples=480, crossfade_samples=240,
                   sola_search_samples=120, device="cpu")
        self.assertEqual(proc.rms_mix._hz_centis, 480)
        self.assertIsNotNone(proc.sola.sola_buffer)

    def test_setup_various_sample_rates(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            proc = AudioProcessor()
            block = sr // 100
            proc.setup(sr=sr, block_samples=block, crossfade_samples=block//2,
                       sola_search_samples=block//4, device="cpu")
            self.assertEqual(proc.rms_mix._hz_centis, sr // 100)
            self.assertIsNotNone(proc.sola.sola_buffer)

    def test_multiple_setups(self):
        proc = AudioProcessor()
        proc.setup(sr=16000, block_samples=160, crossfade_samples=80,
                   sola_search_samples=40, device="cpu")
        self.assertEqual(proc.rms_mix._hz_centis, 160)
        proc.setup(sr=48000, block_samples=480, crossfade_samples=240,
                   sola_search_samples=120, device="cpu")
        self.assertEqual(proc.rms_mix._hz_centis, 480)


class TestAudioProcessorProcessOutput(unittest.TestCase):
    """AudioProcessor process_output 测试。"""

    def setUp(self):
        self.proc = AudioProcessor()
        self.proc.setup(sr=16000, block_samples=160, crossfade_samples=80,
                        sola_search_samples=40, device="cpu")

    def _make_input(self, n=5000):
        return torch.randn(n, dtype=torch.float32) * 0.1

    def test_output_shape(self):
        infer = self._make_input()
        ref = self._make_input()
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertEqual(out.shape, (160,))

    def test_output_finite(self):
        infer = self._make_input()
        ref = self._make_input()
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertTrue(torch.isfinite(out).all())

    def test_is_vc_true_applies_rms(self):
        """is_vc=True 时应用 RMS 混合。"""
        infer = torch.randn(5000, dtype=torch.float32) * 0.05
        ref = torch.randn(5000, dtype=torch.float32) * 0.5
        out = self.proc.process_output(infer, ref, rms_mix=0.0, is_vc=True)
        self.assertTrue(torch.isfinite(out).all())

    def test_is_vc_false_skips_rms(self):
        """is_vc=False 时跳过 RMS 混合，只做 SOLA。"""
        infer = self._make_input()
        ref = self._make_input()
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=False)
        self.assertEqual(out.shape, (160,))
        self.assertTrue(torch.isfinite(out).all())

    def test_rms_mix_one_passthrough_rms(self):
        """rms_mix=1.0 时 RMS 直通，只做 SOLA。"""
        infer = self._make_input()
        ref = self._make_input()
        out = self.proc.process_output(infer, ref, rms_mix=1.0, is_vc=True)
        self.assertEqual(out.shape, (160,))
        self.assertTrue(torch.isfinite(out).all())

    def test_various_rms_mix(self):
        infer = self._make_input()
        ref = self._make_input()
        for mix in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out = self.proc.process_output(infer.clone(), ref, rms_mix=mix, is_vc=True)
            self.assertEqual(out.shape, (160,))
            self.assertTrue(torch.isfinite(out).all())

    def test_consecutive_calls(self):
        for _ in range(10):
            infer = self._make_input()
            ref = self._make_input()
            out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
            self.assertEqual(out.shape, (160,))
            self.assertTrue(torch.isfinite(out).all())

    def test_silence_input(self):
        infer = torch.zeros(5000, dtype=torch.float32)
        ref = self._make_input()
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertTrue(torch.isfinite(out).all())

    def test_silence_reference(self):
        infer = self._make_input()
        ref = torch.zeros(5000, dtype=torch.float32)
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertTrue(torch.isfinite(out).all())


class TestAudioProcessorReset(unittest.TestCase):
    """AudioProcessor reset 测试。"""

    def setUp(self):
        self.proc = AudioProcessor()
        self.proc.setup(sr=16000, block_samples=160, crossfade_samples=80,
                        sola_search_samples=40, device="cpu")

    def test_reset_clears_sola_buffer(self):
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertFalse(torch.allclose(self.proc.sola.sola_buffer, torch.zeros_like(self.proc.sola.sola_buffer)))
        self.proc.reset()
        self.assertTrue(torch.allclose(self.proc.sola.sola_buffer, torch.zeros_like(self.proc.sola.sola_buffer)))

    def test_reset_then_process(self):
        infer = torch.randn(5000, dtype=torch.float32) * 0.1
        ref = torch.randn(5000, dtype=torch.float32) * 0.1
        self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.proc.reset()
        out = self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
        self.assertEqual(out.shape, (160,))
        self.assertTrue(torch.isfinite(out).all())

    def test_multiple_resets(self):
        for _ in range(5):
            infer = torch.randn(5000, dtype=torch.float32) * 0.1
            ref = torch.randn(5000, dtype=torch.float32) * 0.1
            self.proc.process_output(infer, ref, rms_mix=0.5, is_vc=True)
            self.proc.reset()


if __name__ == "__main__":
    unittest.main()
