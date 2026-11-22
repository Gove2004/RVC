"""rvc.dsp.rms — 响度混合的边界语义。"""
import unittest

import torch

from rvc.dsp.rms import apply_rms_mix, fast_rms


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


class TestFastRmsPaddingIsIntentional(unittest.TestCase):
    """锁定 fast_rms 的**块内 padding** 语义，防止被当成 bug「修」掉。

    fast_rms 对每次调用的输入独立做首尾 reflect padding，所以分块算与整段算
    结果不同。这看起来像缺陷，但两条真实路径（实时 runner.process_block、
    离线 offline.run_stream）**都是分块调用且块长相同**，所以不存在
    「实时 ≠ 离线」。若有人为了「分块=整段」而给 fast_rms 加跨块历史拼接，
    本类会失败——那时必须同时确认离线路径也改了，否则反而引入新的不一致。
    """

    SR = 16000

    def test_frame_count_includes_padding_extra_frame(self):
        """帧数 = (N + 2*(frame/2) - frame)//hop + 1，比 N//hop 多 1 帧。

        调用方（apply_rms_mix）靠插值到目标长度消化这个差，所以帧数公式
        是接口的一部分，不是实现细节。
        """
        frame, hop = 4 * (self.SR // 100), self.SR // 100
        for n in (1600, 3200, 8000):
            out = fast_rms(torch.randn(n), frame, hop)
            expected = (n + 2 * (frame // 2) - frame) // hop + 1
            self.assertEqual(out.shape[0], expected, f"n={n} 的帧数公式变了")

    def test_chunked_differs_from_whole_by_design(self):
        """分块与整段**应当**不同（块内 padding 的必然结果）。

        这条断言的作用是「钉住现状」：如果有人给 fast_rms 加了跨块上下文，
        差异会消失，本测试失败并提醒他去看两条真实路径是否都改了。
        """
        frame, hop = 4 * (self.SR // 100), self.SR // 100
        n_total, blk = 32000, 3200
        t = torch.arange(n_total, dtype=torch.float32) / self.SR
        x = (0.3 + 0.2 * torch.sin(2 * torch.pi * 0.7 * t)) * torch.sin(
            2 * torch.pi * 180 * t)

        whole = fast_rms(x, frame, hop)
        chunks = torch.cat([fast_rms(x[i:i + blk], frame, hop)
                            for i in range(0, n_total, blk)])
        # 帧数就不同（每块多算一帧），这本身就是差异来源
        self.assertNotEqual(whole.shape[0], chunks.shape[0])
        m = min(whole.shape[0], chunks.shape[0])
        drift = (whole[:m] - chunks[:m]).abs().max().item()
        self.assertGreater(drift, 1e-4, "分块与整段的差异消失了——"
                                        "若是有意加了跨块上下文，请同步确认离线路径")

    def test_apply_rms_mix_keeps_output_length(self):
        """对外接口不暴露帧数差异：输出长度恒等于输入长度。"""
        frame, hop = 4 * (self.SR // 100), self.SR // 100
        for n in (1600, 3200, 8000):
            ref = torch.randn(n) * 0.2
            conv = torch.randn(n) * 0.5
            out = apply_rms_mix(ref, conv, 0.1, hop)
            self.assertEqual(out.shape[0], n, f"n={n} 的输出长度变了")


if __name__ == "__main__":
    unittest.main()
