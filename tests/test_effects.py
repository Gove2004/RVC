"""音频效果器单元测试 — DenoiseEffect / RmsMixEffect / SolaEffect / AudioProcessor。

全部用 CPU 设备，不依赖 GPU。测试数据用随机张量，验证形状、状态更新和直通逻辑。
"""
import unittest

import torch

from rvc.audio.effects import AudioProcessor, DenoiseEffect, RmsMixEffect, SolaEffect


class TestDenoiseEffect(unittest.TestCase):
    def setUp(self):
        self.effect = DenoiseEffect()
        self.effect.setup(sr=16000)
        self.mono = torch.randn(1024)

    def test_setup_initializes(self):
        self.assertIsNotNone(self.effect._nr_ss)
        self.assertIsNone(self.effect._last_strength)

    def test_process_disabled_returns_same_tensor(self):
        out = self.effect.process(self.mono, enable=False, strength=0.5)
        self.assertIs(out, self.mono)

    def test_process_enabled_returns_same_shape(self):
        out = self.effect.process(self.mono, enable=True, strength=0.5)
        self.assertEqual(out.shape, self.mono.shape)

    def test_strength_change_updates_ss(self):
        self.effect.process(self.mono, enable=True, strength=0.3)
        self.assertEqual(self.effect._last_strength, 0.3)
        self.assertEqual(self.effect._nr_ss.strength, 0.3)
        # 第二次用不同强度，确认更新
        self.effect.process(self.mono, enable=True, strength=0.8)
        self.assertEqual(self.effect._last_strength, 0.8)
        self.assertEqual(self.effect._nr_ss.strength, 0.8)

    def test_reset_clears_noise_floor(self):
        self.effect.process(self.mono, enable=True, strength=0.5)
        self.assertIsNotNone(self.effect._nr_ss.noise_floor)
        self.effect.reset()
        self.assertIsNone(self.effect._nr_ss.noise_floor)


class TestRmsMixEffect(unittest.TestCase):
    def setUp(self):
        self.effect = RmsMixEffect()
        self.effect.setup(sr=16000)
        self.infer = torch.randn(512)
        self.ref = torch.randn(512)

    def test_setup_sets_hz_centis(self):
        self.assertEqual(self.effect._hz_centis, 160)  # 16000 // 100

    def test_full_mix_returns_infer_unchanged(self):
        # rms_mix >= 1.0 时直通，返回原张量
        out = self.effect.process(self.infer, self.ref, rms_mix=1.0)
        self.assertIs(out, self.infer)

    def test_partial_mix_returns_same_shape(self):
        out = self.effect.process(self.infer, self.ref, rms_mix=0.5)
        self.assertEqual(out.shape, self.infer.shape)

    def test_zero_mix_returns_same_shape(self):
        out = self.effect.process(self.infer, self.ref, rms_mix=0.0)
        self.assertEqual(out.shape, self.infer.shape)


class TestSolaEffect(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.block_samples = 256
        self.crossfade_samples = 64
        self.sola_search_samples = 32
        # sola_buffer_samples = min(64, 4*160) = 64
        self.sola_buffer_samples = 64
        self.effect = SolaEffect()
        self.effect.setup(
            sr=self.sr,
            block_samples=self.block_samples,
            crossfade_samples=self.crossfade_samples,
            sola_search_samples=self.sola_search_samples,
            device="cpu",
        )
        # infer 需要至少 block_samples + sola_buffer_samples + sola_search_samples
        self.infer = torch.randn(
            self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        )

    def test_setup_allocates_buffers(self):
        self.assertIsNotNone(self.effect.sola_buffer)
        self.assertEqual(self.effect.sola_buffer.shape, (self.sola_buffer_samples,))
        self.assertIsNotNone(self.effect._fade_in)
        self.assertIsNotNone(self.effect._fade_out)
        self.assertIsNotNone(self.effect._sola_norm_kernel)
        self.assertEqual(self.effect._block_samples, self.block_samples)

    def test_process_returns_block_shape(self):
        out = self.effect.process(self.infer)
        self.assertEqual(out.shape, (self.block_samples,))

    def test_process_updates_sola_buffer(self):
        # 初始 sola_buffer 全零
        self.assertTrue(torch.all(self.effect.sola_buffer == 0))
        self.effect.process(self.infer)
        # 处理后 sola_buffer 被更新为输出块的尾部，不再全零
        self.assertFalse(torch.all(self.effect.sola_buffer == 0))

    def test_reset_clears_sola_buffer(self):
        self.effect.process(self.infer)
        self.assertFalse(torch.all(self.effect.sola_buffer == 0))
        self.effect.reset()
        self.assertTrue(torch.all(self.effect.sola_buffer == 0))

    def test_consecutive_processes_no_crash(self):
        # 连续处理多块，验证状态更新不崩溃
        for _ in range(5):
            infer = torch.randn(
                self.block_samples + self.sola_buffer_samples + self.sola_search_samples
            )
            out = self.effect.process(infer)
            self.assertEqual(out.shape, (self.block_samples,))


class TestAudioProcessor(unittest.TestCase):
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

    def test_setup_initializes_all_effects(self):
        self.assertIsNotNone(self.processor.denoise._nr_ss)
        self.assertEqual(self.processor.rms_mix._hz_centis, 160)
        self.assertIsNotNone(self.processor.sola.sola_buffer)

    def test_process_input_denoise_disabled(self):
        out = self.processor.process_input(self.mono, denoise_enable=False, denoise_strength=0.5)
        self.assertIs(out, self.mono)

    def test_process_input_denoise_enabled(self):
        out = self.processor.process_input(self.mono, denoise_enable=True, denoise_strength=0.5)
        self.assertEqual(out.shape, self.mono.shape)

    def test_process_output_vc_mode(self):
        out = self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=True)
        self.assertEqual(out.shape, (self.block_samples,))

    def test_process_output_passthrough_mode(self):
        # is_vc=False 时跳过 RMS 混合，只做 SOLA
        out = self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=False)
        self.assertEqual(out.shape, (self.block_samples,))

    def test_reset_clears_all_state(self):
        # 先处理一些数据，让状态非零
        self.processor.process_input(self.mono, denoise_enable=True, denoise_strength=0.5)
        self.processor.process_output(self.infer, self.ref, rms_mix=0.5, is_vc=True)
        self.assertIsNotNone(self.processor.denoise._nr_ss.noise_floor)
        self.assertFalse(torch.all(self.processor.sola.sola_buffer == 0))

        self.processor.reset()

        # denoise 噪声地板被清空
        self.assertIsNone(self.processor.denoise._nr_ss.noise_floor)
        # sola buffer 被清零
        self.assertTrue(torch.all(self.processor.sola.sola_buffer == 0))


if __name__ == "__main__":
    unittest.main()
