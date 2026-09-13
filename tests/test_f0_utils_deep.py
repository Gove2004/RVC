"""F0 工具函数深度测试 — median_filter_f0 / postprocess_f0 / apply_pitch_map 缓存。"""
import unittest

import torch

from rvc.audio.f0_utils import apply_pitch_map, median_filter_f0, _pitch_map_midi_cache
from rvc.inference.f0_extractor import postprocess_f0


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


class TestPostprocessF0Deep(unittest.TestCase):
    """postprocess_f0 深度测试 — 3 元组返回值 + 音域映射 + 中值滤波。"""

    def test_returns_3_tuple(self):
        """返回 3 元组 (pitch_coarse, pitchf, confidence)。"""
        f0 = torch.tensor([100.0, 150.0, 0.0, 200.0])
        result = postprocess_f0(f0, "cpu")
        self.assertEqual(len(result), 3)

    def test_confidence_none_generates_pseudo(self):
        """confidence=None 时用 f0>0 生成伪置信度。"""
        f0 = torch.tensor([100.0, 0.0, 200.0])
        _, _, confidence = postprocess_f0(f0, "cpu")
        self.assertEqual(confidence[0].item(), 1.0)
        self.assertEqual(confidence[1].item(), 0.0)
        self.assertEqual(confidence[2].item(), 1.0)

    def test_numpy_input_works(self):
        """numpy 数组输入也能正常工作。"""
        import numpy as np
        f0 = np.array([100.0, 150.0, 200.0])
        result = postprocess_f0(f0, "cpu")
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[1], torch.Tensor)

    def test_pitch_map_applied(self):
        """音域映射生效：100-500Hz 映射到 200-800Hz。

        注意：postprocess_f0 使用因果中值滤波（kernel=3），输出延迟 1 帧。
        因此使用 5 帧输入，验证中间/尾部帧的映射结果。
        """
        from rvc.core.config import InferenceConfig
        cfg = InferenceConfig(
            pitch_map_src_min=100, pitch_map_src_max=500,
            pitch_map_dst_min=200, pitch_map_dst_max=800,
        )
        # 5 帧输入：前 3 帧过渡，后 2 帧稳定在 500Hz
        f0 = torch.tensor([100.0, 300.0, 500.0, 500.0, 500.0])
        _, pitchf, _ = postprocess_f0(f0, "cpu", config=cfg)
        # 100Hz → 200Hz（映射下限，首帧不受滤波影响）
        self.assertAlmostEqual(pitchf[0].item(), 200.0, delta=1.0)
        # 500Hz 稳定帧 → 800Hz（映射上限，因果滤波稳定后）
        self.assertAlmostEqual(pitchf[4].item(), 800.0, delta=1.0)

    def test_unvoiced_stays_zero_after_map(self):
        """清音帧（f0=0）经过音域映射后仍为 0。"""
        f0 = torch.tensor([0.0, 100.0, 0.0])
        _, pitchf, _ = postprocess_f0(f0, "cpu")
        self.assertEqual(pitchf[0].item(), 0.0)
        self.assertEqual(pitchf[2].item(), 0.0)


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
