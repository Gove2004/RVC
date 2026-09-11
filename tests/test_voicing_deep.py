"""清浊分析深度测试 — causal_median_filter / causal_moving_average / compute_uv_prob。"""
import unittest

import torch

from rvc.inference.voicing import causal_median_filter, causal_moving_average, compute_uv_prob


class TestCausalMedianFilter(unittest.TestCase):
    """因果中值滤波测试。"""

    def test_causal_not_future(self):
        """因果性：输出只依赖当前和过去帧，不依赖未来帧。"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out1 = causal_median_filter(x, kernel_size=3)
        # 改变未来帧（最后一帧），前 N-1 帧输出应不变
        x2 = x.clone()
        x2[0, -1] = 100.0
        out2 = causal_median_filter(x2, kernel_size=3)
        self.assertTrue(torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5))

    def test_outlier_removed(self):
        """连续帧中的离群值被中值滤波去除。"""
        x = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        out = causal_median_filter(x, kernel_size=3)
        # 中间的 1.0 是离群值，中值应为 0
        self.assertEqual(out[0, 2].item(), 0.0)

    def test_step_boundary_preserved(self):
        """阶跃边界被保留（中值滤波非线性，不模糊真实边界）。"""
        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        out = causal_median_filter(x, kernel_size=3)
        # 边界帧（索引 3）= median(0, 0, 1) = 0，边界后一帧 = median(0, 1, 1) = 1
        self.assertEqual(out[0, 3].item(), 0.0)
        self.assertEqual(out[0, 4].item(), 1.0)

    def test_constant_unchanged(self):
        """常数输入输出不变。"""
        x = torch.ones(1, 10) * 0.5
        out = causal_median_filter(x, kernel_size=3)
        self.assertTrue(torch.allclose(out, x, atol=1e-5))

    def test_output_shape(self):
        """输出形状与输入相同。"""
        for shape in [(1, 10), (2, 5), (3, 20)]:
            x = torch.randn(*shape)
            out = causal_median_filter(x, kernel_size=3)
            self.assertEqual(out.shape, x.shape)

    def test_kernel_1_returns_unchanged(self):
        """kernel_size=1 时原样返回。"""
        x = torch.randn(1, 10)
        out = causal_median_filter(x, kernel_size=1)
        self.assertTrue(torch.allclose(out, x))


class TestCausalMovingAverage(unittest.TestCase):
    """因果移动平均测试。"""

    def test_step_response(self):
        """阶跃响应：边界帧平滑过渡。"""
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        out = causal_moving_average(x, kernel_size=3)
        # 边界帧（索引 5）= (1 + 0 + 0) / 3 = 1/3
        self.assertAlmostEqual(out[0, 5].item(), 1.0 / 3.0, places=4)
        # 边界后一帧（索引 6）= (1 + 1 + 0) / 3 = 2/3
        self.assertAlmostEqual(out[0, 6].item(), 2.0 / 3.0, places=4)
        # 边界后两帧（索引 7）= (1 + 1 + 1) / 3 = 1
        self.assertAlmostEqual(out[0, 7].item(), 1.0, places=4)

    def test_causal_not_future(self):
        """因果性：改变未来帧不影响过去输出。"""
        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        out1 = causal_moving_average(x, kernel_size=3)
        x2 = x.clone()
        x2[0, -1] = 100.0
        out2 = causal_moving_average(x2, kernel_size=3)
        self.assertTrue(torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5))

    def test_constant_unchanged(self):
        """常数输入输出不变。"""
        x = torch.ones(1, 10) * 0.5
        out = causal_moving_average(x, kernel_size=3)
        self.assertTrue(torch.allclose(out, x, atol=1e-5))

    def test_output_shape(self):
        """输出形状与输入相同。"""
        x = torch.randn(2, 5, 10)
        out = causal_moving_average(x, kernel_size=3)
        self.assertEqual(out.shape, x.shape)

    def test_kernel_1_returns_unchanged(self):
        """kernel_size=1 时原样返回。"""
        x = torch.randn(1, 10)
        out = causal_moving_average(x, kernel_size=1)
        self.assertTrue(torch.allclose(out, x))


class TestComputeUVProb(unittest.TestCase):
    """compute_uv_prob 清浊概率计算测试。"""

    def test_high_pitch_low_uv(self):
        """高音高（浊音）→ uv_prob 低。"""
        pitchf = torch.ones(1, 10) * 200  # 200Hz，明显浊音
        confidence = torch.ones(1, 10) * 0.9
        uv = compute_uv_prob(pitchf, confidence)
        # 中心帧应该接近 0（浊音）
        self.assertLess(uv[0, 5].item(), 0.1)

    def test_zero_pitch_high_uv(self):
        """零音高（清音/UV）→ uv_prob 高。"""
        pitchf = torch.zeros(1, 10)
        confidence = torch.zeros(1, 10)
        uv = compute_uv_prob(pitchf, confidence)
        # sigmoid((20-0)/(30/4)) = sigmoid(2.67) ≈ 0.935
        self.assertGreater(uv[0, 5].item(), 0.8)

    def test_low_confidence_increases_uv(self):
        """低 confidence 但 F0 非零 → uv_prob 提高（confidence 修正生效）。"""
        pitchf = torch.ones(1, 10) * 100  # 100Hz，F0 基 uv_prob 低
        conf_high = torch.ones(1, 10) * 0.9
        conf_low = torch.ones(1, 10) * 0.05  # 低 confidence
        uv_high = compute_uv_prob(pitchf, conf_high)
        uv_low = compute_uv_prob(pitchf, conf_low)
        # 低 confidence 的 uv_prob 应该更高
        self.assertGreater(uv_low[0, 5].item(), uv_high[0, 5].item())

    def test_conf_threshold_linkage(self):
        """conf_threshold 联动：调高阈值 → confidence 修正中心提高 → uv_prob 提高。"""
        pitchf = torch.ones(1, 10) * 100
        confidence = torch.ones(1, 10) * 0.07
        uv_low = compute_uv_prob(pitchf, confidence, conf_threshold=0.05)
        uv_high = compute_uv_prob(pitchf, confidence, conf_threshold=0.08)
        self.assertGreater(uv_high[0, 5].item(), uv_low[0, 5].item())

    def test_output_range(self):
        """输出在 [0, 1] 范围内。"""
        pitchf = torch.rand(1, 20) * 500
        confidence = torch.rand(1, 20)
        uv = compute_uv_prob(pitchf, confidence)
        self.assertTrue(torch.all(uv >= 0))
        self.assertTrue(torch.all(uv <= 1))

    def test_output_shape(self):
        """输出形状与输入相同。"""
        pitchf = torch.randn(1, 15)
        confidence = torch.randn(1, 15)
        uv = compute_uv_prob(pitchf, confidence)
        self.assertEqual(uv.shape, pitchf.shape)

    def test_smoothing_reduces_jitter(self):
        """时间平滑减少逐帧抖动（相邻帧差异减小）。"""
        # 构造有抖动的输入
        pitchf = torch.tensor([[100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]])
        confidence = torch.ones(1, 10) * 0.9
        uv = compute_uv_prob(pitchf, confidence)
        # 平滑后相邻帧差异应该小于原始 sigmoid 的差异
        raw_uv = torch.sigmoid((20.0 - pitchf) / (30.0 / 4))
        raw_diff = torch.abs(torch.diff(raw_uv[0])).mean()
        smooth_diff = torch.abs(torch.diff(uv[0])).mean()
        self.assertLess(smooth_diff.item(), raw_diff.item())

    def test_f0_zero_confidence_high(self):
        """F0=0 但 confidence 高 → uv_prob 由 F0 基决定（接近 1）。"""
        pitchf = torch.zeros(1, 10)
        confidence = torch.ones(1, 10) * 0.9  # 高 confidence 但 F0=0
        uv = compute_uv_prob(pitchf, confidence)
        # F0=0 时 uv_conf 被乘 0（pitchf>0 的 mask），所以 uv_prob 由 uv_f0 决定
        self.assertGreater(uv[0, 5].item(), 0.8)


if __name__ == "__main__":
    unittest.main()
