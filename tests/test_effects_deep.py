"""音频效果器深度测试 — 边界条件 / 状态一致性 / 连续处理 / 参数极端值。

在 test_effects.py 基础上补充更深入的测试用例。
全部用 CPU 设备，不依赖 GPU。
"""
import unittest

import torch

from rvc.audio.effects import AudioProcessor, DenoiseEffect, RmsMixEffect, SolaEffect


class TestDenoiseEffectDeep(unittest.TestCase):
    """DenoiseEffect 边界条件和状态测试。"""

    def setUp(self):
        self.effect = DenoiseEffect()
        self.effect.setup(sr=16000)
        self.mono = torch.randn(1024)

    def test_strength_zero_still_processes(self):
        """strength=0 时仍会处理（不是直通），只是降噪强度为 0。"""
        out = self.effect.process(self.mono, enable=True, strength=0.0)
        self.assertEqual(out.shape, self.mono.shape)
        # strength=0 时 _last_strength 会更新
        self.assertEqual(self.effect._last_strength, 0.0)

    def test_strength_negative_clamped_by_ss(self):
        """负 strength 传给 SpectralSubtraction，不崩溃。"""
        out = self.effect.process(self.mono, enable=True, strength=-0.5)
        self.assertEqual(out.shape, self.mono.shape)

    def test_strength_above_one(self):
        """strength>1 时不崩溃。"""
        out = self.effect.process(self.mono, enable=True, strength=1.5)
        self.assertEqual(out.shape, self.mono.shape)

    def test_same_strength_no_ss_update(self):
        """相同 strength 连续调用不会重复 set_strength。"""
        self.effect.process(self.mono, enable=True, strength=0.5)
        first_ss = self.effect._nr_ss
        self.effect.process(self.mono, enable=True, strength=0.5)
        # _last_strength 不变，不会触发 set_strength
        self.assertEqual(self.effect._last_strength, 0.5)
        self.assertIs(self.effect._nr_ss, first_ss)

    def test_disabled_after_enabled_no_state_change(self):
        """禁用后再启用，状态保持。"""
        self.effect.process(self.mono, enable=True, strength=0.5)
        self.effect.process(self.mono, enable=False, strength=0.8)
        # 禁用时不更新 _last_strength
        self.assertEqual(self.effect._last_strength, 0.5)


class TestRmsMixEffectDeep(unittest.TestCase):
    """RmsMixEffect 边界条件测试。"""

    def setUp(self):
        self.effect = RmsMixEffect()
        self.effect.setup(sr=16000)
        self.infer = torch.randn(512)
        self.ref = torch.randn(512)

    def test_rms_mix_above_one_passthrough(self):
        """rms_mix > 1.0 时直通。"""
        out = self.effect.process(self.infer, self.ref, rms_mix=1.5)
        self.assertIs(out, self.infer)

    def test_rms_mix_exactly_one_passthrough(self):
        """rms_mix == 1.0 时直通。"""
        out = self.effect.process(self.infer, self.ref, rms_mix=1.0)
        self.assertIs(out, self.infer)

    def test_different_sr_hz_centis(self):
        """不同采样率下 _hz_centis 正确。"""
        effect = RmsMixEffect()
        effect.setup(sr=48000)
        self.assertEqual(effect._hz_centis, 480)  # 48000 // 100


class TestSolaEffectDeep(unittest.TestCase):
    """SolaEffect 边界条件和状态一致性测试。"""

    def _make_effect(self, sr=16000, block_samples=256, crossfade_samples=64,
                      sola_search_samples=32):
        effect = SolaEffect()
        effect.setup(sr, block_samples, crossfade_samples, sola_search_samples, "cpu")
        return effect

    def test_sola_buffer_samples_capped(self):
        """sola_buffer_samples = min(crossfade_samples, 4*zc)，crossfade 过大时被截断。"""
        effect = self._make_effect(crossfade_samples=1000)  # 远大于 4*160=640
        self.assertEqual(effect._sola_buffer_samples, 640)  # 被截断到 4*zc

    def test_small_crossfade(self):
        """很小的 crossfade 不崩溃。"""
        effect = self._make_effect(crossfade_samples=8, sola_search_samples=4)
        infer = torch.randn(256 + 8 + 4)
        out = effect.process(infer)
        self.assertEqual(out.shape, (256,))

    def test_fade_in_out_sum_to_one(self):
        """fade_in + fade_out = 1。"""
        effect = self._make_effect()
        self.assertTrue(torch.allclose(effect._fade_in + effect._fade_out, torch.ones_like(effect._fade_in)))

    def test_fade_in_monotonic_increasing(self):
        """fade_in 单调递增。"""
        effect = self._make_effect()
        diffs = effect._fade_in[1:] - effect._fade_in[:-1]
        self.assertTrue(torch.all(diffs >= 0))

    def test_reset_after_multiple_blocks(self):
        """多块处理后 reset 清空 buffer。"""
        effect = self._make_effect()
        for _ in range(10):
            infer = torch.randn(256 + 64 + 32)
            effect.process(infer)
        self.assertFalse(torch.all(effect.sola_buffer == 0))
        effect.reset()
        self.assertTrue(torch.all(effect.sola_buffer == 0))

    def test_deterministic_with_same_input(self):
        """相同输入和初始状态，输出一致。"""
        effect1 = self._make_effect()
        effect2 = self._make_effect()
        infer = torch.randn(256 + 64 + 32)
        out1 = effect1.process(infer)
        out2 = effect2.process(infer)
        self.assertTrue(torch.allclose(out1, out2))


class TestAudioProcessorDeep(unittest.TestCase):
    """AudioProcessor 集成测试 — 效果器链顺序和状态一致性。"""

    def setUp(self):
        self.sr = 16000
        self.block_samples = 256
        self.crossfade_samples = 64
        self.sola_search_samples = 32
        self.sola_buffer_samples = 64
        self.processor = AudioProcessor()
        self.processor.setup(
            sr=self.sr,
            block_samples=self.block_samples,
            crossfade_samples=self.crossfade_samples,
            sola_search_samples=self.sola_search_samples,
            device="cpu",
        )
        self.mono = torch.randn(1024)
        self.infer = torch.randn(
            self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        )
        self.ref = torch.randn(
            self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        )

    def test_process_chain_order_vc_mode(self):
        """VC 模式下：RMS 混合 → SOLA，输出长度为 block_samples。"""
        out = self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=True)
        self.assertEqual(out.shape, (self.block_samples,))

    def test_process_chain_order_passthrough_mode(self):
        """直通模式下：跳过 RMS，只做 SOLA。"""
        out = self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=False)
        self.assertEqual(out.shape, (self.block_samples,))

    def test_full_pipeline_continuous(self):
        """完整 pipeline 连续处理 20 块不崩溃，状态持续更新。"""
        for i in range(20):
            mono = torch.randn(1024)
            infer = torch.randn(self.block_samples + self.sola_buffer_samples + self.sola_search_samples)
            ref = torch.randn(self.block_samples + self.sola_buffer_samples + self.sola_search_samples)

            processed_input = self.processor.process_input(mono, denoise_enable=True, denoise_strength=0.5)
            self.assertEqual(processed_input.shape, mono.shape)

            out = self.processor.process_output(infer, ref, rms_mix=0.5, is_vc=True)
            self.assertEqual(out.shape, (self.block_samples,))

        # 连续处理后 sola_buffer 非零
        self.assertFalse(torch.all(self.processor.sola.sola_buffer == 0))

    def test_reset_between_sessions(self):
        """reset 后状态清空，新会话不受旧会话影响。"""
        # 会话 1
        for _ in range(5):
            self.processor.process_input(self.mono, denoise_enable=True, denoise_strength=0.5)
            self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=True)

        self.processor.reset()

        # 会话 2 开始时状态干净
        self.assertIsNone(self.processor.denoise._nr_ss.noise_floor)
        self.assertTrue(torch.all(self.processor.sola.sola_buffer == 0))

    def test_denoise_strength_runtime_change(self):
        """运行中改变降噪强度不崩溃。"""
        for strength in [0.0, 0.3, 0.5, 0.8, 1.0]:
            out = self.processor.process_input(self.mono, denoise_enable=True, denoise_strength=strength)
            self.assertEqual(out.shape, self.mono.shape)

    def test_rms_mix_runtime_change(self):
        """运行中改变 rms_mix 不崩溃。"""
        for rms_mix in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]:
            out = self.processor.process_output(self.infer, self.ref, rms_mix=rms_mix, is_vc=True)
            self.assertEqual(out.shape, (self.block_samples,))


if __name__ == "__main__":
    unittest.main()
