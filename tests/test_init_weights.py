"""init_weights 必须真正落地 — weight_norm 参数化后写 m.weight 会被丢弃。

WN 卷积的真实参数是 parametrizations.weight.original0；init_weights 若写
计算出来的 m.weight，normal_() 写进临时张量即弃，从零训练的初始化静默失效。
"""
import unittest

import torch
from torch.nn.utils import parametrizations

from rvc.nn.vits_blocks import init_weights


class TestInitWeightsLands(unittest.TestCase):
    def test_weight_norm_conv_initializes_original0(self):
        conv = torch.nn.Conv1d(8, 8, 3)
        parametrizations.weight_norm(conv, name="weight")
        before = conv.parametrizations.weight.original0.detach().clone()
        init_weights(conv, mean=0.0, std=0.01)
        after = conv.parametrizations.weight.original0
        self.assertFalse(torch.equal(before, after), "original0 必须被改写")
        self.assertLess(float(after.std()), 0.05, "std 应接近 0.01 而非默认初始化")
        self.assertLess(abs(float(after.mean())), 0.02)

    def test_plain_conv_still_initializes(self):
        conv = torch.nn.Conv1d(8, 8, 3)
        before = conv.weight.detach().clone()
        init_weights(conv, mean=0.0, std=0.01)
        self.assertFalse(torch.equal(before, conv.weight))
        self.assertLess(float(conv.weight.std()), 0.05)

    def test_non_conv_is_untouched(self):
        linear = torch.nn.Linear(8, 8)
        before = linear.weight.detach().clone()
        init_weights(linear)
        self.assertTrue(torch.equal(before, linear.weight))


if __name__ == "__main__":
    unittest.main()