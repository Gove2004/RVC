"""rvc.dsp.rms — 响度混合的边界语义。"""
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

    def test_long_reference_aligns_to_tail(self):
        """参考窗比转换块长时（crossfade>100ms，sola_buffer 被截到 100ms），
        转换块对应输入窗的**最新** n 个样本，必须取参考尾部 [-n:]。

        取 [:n] 会错位到最旧的一段：这里头部响、尾部安静，错位会让输出跟随
        头部响度而非时间对齐的尾部。与「参考恰为尾部」的对照须逐样本一致。
        """
        spc = 160
        head = make_signal(sr=16000, sec=0.2, amp=0.5)   # 响（最旧）
        tail = make_signal(sr=16000, sec=0.2, amp=0.02)  # 安静（最新，与转换对齐）
        conv = make_signal(sr=16000, sec=0.2, amp=0.3)
        ref = torch.cat([head, tail])
        out = apply_rms_mix(ref, conv, rms_mix=0.0, samples_per_centis=spc)
        ctrl = apply_rms_mix(tail, conv, rms_mix=0.0, samples_per_centis=spc)
        self.assertTrue(torch.allclose(out, ctrl, atol=1e-6),
                        "长参考必须与「参考=尾部」逐样本一致（取 [-n:] 而非 [:n]）")
        out_rms = float(out.pow(2).mean().sqrt())
        head_rms = float(head.pow(2).mean().sqrt())
        tail_rms = float(tail.pow(2).mean().sqrt())
        self.assertLess(out_rms, (head_rms + tail_rms) / 2, "输出应跟随安静的尾部参考")


if __name__ == "__main__":
    unittest.main()
