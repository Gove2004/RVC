"""rvc.dsp.rms — 响度混合与输入噪声门的边界语义。"""
import unittest

import torch

from rvc.dsp.rms import apply_noise_gate, apply_rms_mix


def make_signal(sr=16000, sec=0.5, amp=0.5):
    t = torch.arange(int(sr * sec), dtype=torch.float32) / sr
    return amp * torch.sin(2 * torch.pi * 220 * t)


class TestRmsMix(unittest.TestCase):
    def test_mix_one_keeps_converted(self):
        ref, conv = make_signal(amp=0.2), make_signal(amp=0.5)
        out = apply_rms_mix(ref, conv, rms_mix=1.0, samples_per_centis=160)
        self.assertTrue(torch.allclose(out, conv, atol=1e-6))

    def test_mix_zero_follows_reference_envelope(self):
        # 参考很安静、转换很响：rms_mix=0 时输出应被压到接近参考电平
        ref, conv = make_signal(amp=0.02), make_signal(amp=0.5)
        out = apply_rms_mix(ref, conv, rms_mix=0.0, samples_per_centis=160)
        rms_out = float(out.pow(2).mean().sqrt())
        rms_ref = float(ref.pow(2).mean().sqrt())
        self.assertLess(rms_out, rms_ref * 1.5, "输出应跟随参考响度")

    def test_output_length_preserved(self):
        ref, conv = make_signal(), make_signal()
        out = apply_rms_mix(ref, conv, rms_mix=0.5, samples_per_centis=160)
        self.assertEqual(out.shape[0], conv.shape[0])


class TestNoiseGate(unittest.TestCase):
    """输入噪声门：阈值即门限，低于阈值 dB 的帧静音（参考原版响应阈值）。"""

    def test_loud_signal_passes_unchanged(self):
        # amp=0.5 正弦 RMS≈-9dB，远高于 -60dB 门限 → 不该被门控
        wav = make_signal(amp=0.5)
        out = apply_noise_gate(wav, threshold_db=-60.0, samples_per_centis=160)
        self.assertTrue(torch.allclose(out, wav, atol=1e-6))

    def test_quiet_signal_gated_to_zero(self):
        # amp=1e-4 正弦 RMS≈-83dB，低于 -60dB 门限 → 全帧静音
        wav = make_signal(amp=1e-4)
        out = apply_noise_gate(wav, threshold_db=-60.0, samples_per_centis=160)
        self.assertTrue(torch.all(out == 0.0), "低于门限的安静信号应被完全静音")

    def test_min_threshold_lets_quiet_signal_pass(self):
        # -100dB 近似关闭：-83dB 信号高于 -100dB 门限 → 不门控
        wav = make_signal(amp=1e-4)
        out = apply_noise_gate(wav, threshold_db=-100.0, samples_per_centis=160)
        self.assertTrue(torch.allclose(out, wav, atol=1e-6))

    def test_only_quiet_frames_gated(self):
        # 前响后轻：只静音安静段，响亮段保留（避开拼接边界帧）
        loud = make_signal(amp=0.5, sec=0.25)
        quiet = make_signal(amp=1e-4, sec=0.25)
        wav = torch.cat([loud, quiet])
        out = apply_noise_gate(wav, threshold_db=-60.0, samples_per_centis=160)
        half = loud.shape[0] // 2
        self.assertTrue(torch.allclose(out[:half], loud[:half], atol=1e-6))
        self.assertTrue(torch.all(out[-half:] == 0.0))

    def test_output_length_preserved(self):
        wav = make_signal()
        out = apply_noise_gate(wav, threshold_db=-60.0, samples_per_centis=160)
        self.assertEqual(out.shape[0], wav.shape[0])

    def test_input_not_mutated(self):
        # mono 是持久上传缓冲的视图：门控必须返回新张量，不改写入参
        wav = make_signal(amp=1e-4)
        snapshot = wav.clone()
        apply_noise_gate(wav, threshold_db=-60.0, samples_per_centis=160)
        self.assertTrue(torch.equal(wav, snapshot))


if __name__ == "__main__":
    unittest.main()
