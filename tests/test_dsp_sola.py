"""SOLA 对齐纯函数测试 — S3 新增覆盖（原 streaming/alignment.py 无测试）。

核心约定（重写必须保持）：
- 完美对齐（buffer 与输入头部一致）时 offset=0；
- 交叉淡化输出 = infer头部*fade_in + buffer*fade_out；
- sola_buffer 更新为输出块之后的尾部；
- NaN 污染防护：buffer 含 NaN 时清零 buffer、跳过淡化直接返回。
"""
import unittest

import torch

from rvc.dsp.sola import apply_sola


def _make_args(block=100, buf=20, search=50):
    infer = torch.sin(torch.linspace(0, 40, block + search)) * 0.3
    sola_buffer = infer[:buf].clone()
    norm_kernel = torch.ones(1, 1, buf)  # 与真实调用方一致：3D 核
    fade_in = torch.ones(buf)
    fade_out = torch.zeros(buf)  # 淡出为 0 → 交叉淡化结果 = infer 头部原样
    return infer, sola_buffer, norm_kernel, fade_in, fade_out, block, buf, search


class TestApplySola(unittest.TestCase):
    def test_perfect_alignment_offset_zero(self):
        args = _make_args()
        out = apply_sola(*args)
        assert out.shape[0] == 100
        # fade_out=0 → 输出头部 = infer 原样（未混合旧 buffer）
        torch.testing.assert_close(out, args[0][:100])

    def test_buffer_updated_with_tail(self):
        infer, buf, *rest = _make_args()
        apply_sola(infer, buf, *rest)
        # 新 buffer = 输出块之后的尾部 [block:block+buf]
        # offset=0 时 infer 未切片，尾部即 infer[100:120]
        torch.testing.assert_close(buf, infer[100:120])

    def test_nan_buffer_skips_crossfade(self):
        infer, buf, norm, fade_in, fade_out, block, b, s = _make_args()
        buf[5] = float("nan")
        out = apply_sola(infer, buf, norm, fade_in, fade_out, block, b, s)
        assert not torch.isnan(out).any()
        assert torch.all(buf == 0)  # 被清零
        # 跳过淡化：直接返回偏移后的块

    def test_nan_tail_clears_buffer(self):
        block, buf_n, search = 100, 20, 50
        infer = torch.zeros(block + search)
        infer[105] = float("nan")  # 落在尾部区
        buf = torch.zeros(buf_n)
        norm = torch.ones(1, 1, buf_n)
        out = apply_sola(
            infer, buf, norm, torch.ones(buf_n), torch.zeros(buf_n),
            block, buf_n, search,
        )
        assert torch.all(buf == 0)


if __name__ == "__main__":
    unittest.main()
