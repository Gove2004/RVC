"""rvc.dsp.rms.apply_rms_mix — 响度混合边界语义。"""
import unittest

import torch

from rvc.dsp.rms import apply_rms_mix


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


if __name__ == "__main__":
    unittest.main()
