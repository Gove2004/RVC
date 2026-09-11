"""F0 工具函数与清浊分析深度测试 — median_filter_f0 / causal_moving_average / postprocess_f0 / apply_pitch_map 缓存。"""
import unittest

import torch

from rvc.audio.f0_utils import apply_pitch_map, median_filter_f0, _pitch_map_midi_cache
from rvc.inference.f0_extractor import postprocess_f0
from rvc.inference.voicing import causal_moving_average, compute_uv_prob


class TestMedianFilterF0(unittest.TestCase):
    """median_filter_f0 因果中值滤波测试。"""

    def test_causal_not_future(self):
        """因果性：输出只依赖当前和过去帧，不依赖未来帧。"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out1 = median_filter_f0(x, kernel=3)
        # 改变未来帧（最后一帧），前 N-1 帧输出应不变
        x2 = x.clone()
        x2[-1] = 100.0
        out2 = median_filter_f0(x2, kernel=3)
        self.assertTrue(torch.allclose(out1[:-1], out2[:-1], atol=1e-5))

    def test_unvoiced_stays_zero(self):
        """清音帧（f0=0）始终保持 0。"""
        x = torch.zeros(10)
        out = median_filter_f0(x, kernel=3)
        self.assertTrue(torch.all(out == 0))

    def test_voiced_filtered_to_zero_restores(self):
        """浊音帧被滤波成 0 时（窗口零值占多数），恢复原值避免误清。"""
        # 中间一个浊音帧被清音包围
        x = torch.tensor([0.0, 0.0, 150.0, 0.0, 0.0])
        out = median_filter_f0(x, kernel=3)
        # 中间帧中值应为 0（左右都是 0），但浊音被滤波成 0 时恢复原值
        self.assertEqual(out[2].item(), 150.0)

    def test_outlier_removed(self):
        """连续浊音帧中的离群值被中值滤波去除。"""
        x = torch.tensor([100.0, 100.0, 500.0, 100.0, 100.0])
        out = median_filter_f0(x, kernel=3)
        # 中间的 500 是离群值，中值应为 100
        self.assertEqual(out[2].item(), 100.0)

    def test_output_shape(self):
        """输出形状与输入相同。"""
        for n in [3, 5, 10, 100]:
            x = torch.randn(n).abs() + 10
            out = median_filter_f0(x, kernel=3)
            self.assertEqual(out.shape, x.shape)

    def test_short_input_returns_unchanged(self):
        """输入长度小于 kernel 时原样返回。"""
        x = torch.tensor([100.0, 100.0])
        out = median_filter_f0(x, kernel=3)
        self.assertTrue(torch.allclose(out, x))

    def test_non_tensor_returns_unchanged(self):
        """非 tensor 输入原样返回。"""
        x = [100.0, 200.0, 300.0]
        out = median_filter_f0(x, kernel=3)
        self.assertEqual(out, x)


class TestCausalMovingAverage(unittest.TestCase):
    """causal_moving_average 因果移动平均测试。"""

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
        x2[0, -1] = 100.0  # 改变未来帧
        out2 = causal_moving_average(x2, kernel_size=3)
        # 前 N-1 帧输出应不变（只看当前和过去 2 帧）
        self.assertTrue(torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5))

    def test_constant_input_unchanged(self):
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


class TestPostprocessF0Deep(unittest.TestCase):
    """postprocess_f0 深度测试 — 4 元组返回值 + 清音帧保护。"""

    def test_returns_4_tuple(self):
        """返回 4 元组 (pitch_coarse, pitchf, confidence, f0_raw)。"""
        f0 = torch.tensor([100.0, 150.0, 0.0, 200.0])
        result = postprocess_f0(f0, "cpu")
        self.assertEqual(len(result), 4)

    def test_f0_raw_equals_original(self):
        """f0_raw 等于原始输入（音域映射之前）。"""
        f0 = torch.tensor([100.0, 150.0, 0.0, 200.0])
        _, _, _, f0_raw = postprocess_f0(f0, "cpu")
        self.assertTrue(torch.allclose(f0_raw, f0, atol=1e-5))

    def test_f0_raw_unvoiced_stays_zero(self):
        """f0_raw 中清音帧保持 0。"""
        f0 = torch.tensor([0.0, 0.0, 100.0, 0.0])
        _, _, _, f0_raw = postprocess_f0(f0, "cpu")
        self.assertTrue(torch.all(f0_raw[[0, 1, 3]] == 0))

    def test_low_f0_protected_as_unvoiced(self):
        """低 F0 帧（低于保护阈值）被设为 0，避免假 F0 送入合成器。"""
        # 保护阈值 = protect_soft_threshold_hz + protect_soft_width / 2 = 20 + 15 = 35Hz
        # 30Hz 低于 35Hz，应被保护为 0
        f0 = torch.tensor([30.0, 100.0, 200.0])
        _, pitchf, _, _ = postprocess_f0(f0, "cpu")
        self.assertEqual(pitchf[0].item(), 0.0)
        # 100Hz 和 200Hz 高于阈值，应保留（经过音域映射）
        self.assertGreater(pitchf[1].item(), 0.0)
        self.assertGreater(pitchf[2].item(), 0.0)

    def test_confidence_none_generates_pseudo(self):
        """confidence=None 时用 f0>0 生成伪置信度。"""
        f0 = torch.tensor([100.0, 0.0, 200.0])
        _, _, confidence, _ = postprocess_f0(f0, "cpu")
        self.assertEqual(confidence[0].item(), 1.0)
        self.assertEqual(confidence[1].item(), 0.0)
        self.assertEqual(confidence[2].item(), 1.0)

    def test_numpy_input_works(self):
        """numpy 数组输入也能正常工作。"""
        import numpy as np
        f0 = np.array([100.0, 150.0, 200.0])
        result = postprocess_f0(f0, "cpu")
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[1], torch.Tensor)


class TestComputeUVProbConfThreshold(unittest.TestCase):
    """compute_uv_prob 的 conf_threshold 参数联动测试。"""

    def test_default_matches_hardcoded(self):
        """默认 conf_threshold=0.05 与原硬编码 sigmoid((0.1-conf)/0.03) 一致。"""
        pitchf = torch.tensor([[100.0, 100.0, 100.0]])
        confidence = torch.tensor([[0.07, 0.07, 0.07]])
        uv = compute_uv_prob(pitchf, confidence, conf_threshold=0.05)
        expected = torch.sigmoid(torch.tensor((0.1 - 0.07) / 0.03))
        # 经过中值滤波和移动平均后，值会略有变化，但中心帧应接近
        self.assertAlmostEqual(uv[0, 1].item(), expected.item(), delta=0.1)

    def test_higher_threshold_increases_uv(self):
        """调高 conf_threshold → confidence 修正中心提高 → uv 概率提高。"""
        pitchf = torch.tensor([[100.0] * 10])
        confidence = torch.tensor([[0.07] * 10])
        uv_low = compute_uv_prob(pitchf, confidence, conf_threshold=0.05)
        uv_high = compute_uv_prob(pitchf, confidence, conf_threshold=0.08)
        # 调高阈值后，conf=0.07 更低于中心，uv 应更高
        self.assertGreater(uv_high[0, 5].item(), uv_low[0, 5].item())

    def test_zero_confidence_high_uv(self):
        """confidence=0 时 uv 概率应接近 1。"""
        pitchf = torch.tensor([[100.0] * 5])
        confidence = torch.zeros(1, 5)
        uv = compute_uv_prob(pitchf, confidence, conf_threshold=0.05)
        self.assertGreater(uv[0, 2].item(), 0.9)


class TestApplyPitchMapCache(unittest.TestCase):
    """apply_pitch_map 预计算缓存测试。"""

    def setUp(self):
        # 清空缓存
        _pitch_map_midi_cache.clear()

    def test_cache_populated_after_call(self):
        """调用后缓存被填充。"""
        f0 = torch.tensor([100.0, 200.0, 300.0])
        apply_pitch_map(f0, 100, 500, 200, 800)
        self.assertEqual(len(_pitch_map_midi_cache), 1)

    def test_same_params_cache_hit(self):
        """相同参数多次调用使用缓存，结果一致。"""
        f0 = torch.tensor([100.0, 200.0, 300.0])
        out1 = apply_pitch_map(f0, 100, 500, 200, 800)
        out2 = apply_pitch_map(f0, 100, 500, 200, 800)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-4))
        # 缓存只有 1 条（相同参数命中）
        self.assertEqual(len(_pitch_map_midi_cache), 1)

    def test_different_params_separate_cache(self):
        """不同参数使用不同缓存条目。"""
        f0 = torch.tensor([100.0, 200.0, 300.0])
        apply_pitch_map(f0, 100, 500, 200, 800)
        apply_pitch_map(f0, 80, 300, 200, 500)
        self.assertEqual(len(_pitch_map_midi_cache), 2)

    def test_cache_size_limited(self):
        """缓存大小有限制（超过 128 条时淘汰旧条目）。"""
        f0 = torch.tensor([100.0, 200.0])
        for i in range(130):
            apply_pitch_map(f0, 100 + i, 500 + i, 200 + i, 800 + i)
        self.assertLessEqual(len(_pitch_map_midi_cache), 128)

    def test_cached_result_matches_uncached(self):
        """缓存结果与无缓存结果一致（验证缓存正确性）。"""
        f0 = torch.tensor([100.0, 150.0, 200.0, 0.0, 300.0])
        # 第一次调用（无缓存）
        _pitch_map_midi_cache.clear()
        out1 = apply_pitch_map(f0, 100, 500, 200, 800)
        # 清空缓存后再次调用（重新计算）
        _pitch_map_midi_cache.clear()
        out2 = apply_pitch_map(f0, 100, 500, 200, 800)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
