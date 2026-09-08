"""特征处理深度测试 — clone_protect_source/protect_blend/upsample_features。

覆盖：
- clone_protect_source: use_f0/protect 各种组合
- protect_blend: 浊音/清音/各种 protect 值/软阈值模式
- upsample_features: 各种输入形状/p_len/is_half/feats0/pitchf
- 注意：extract_hubert_features 需要真实模型，只测试不调用真实模型的部分
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from rvc.inference.feature_processing import (
    clone_protect_source,
    protect_blend,
    upsample_features,
)


class TestCloneProtectSource(unittest.TestCase):
    """clone_protect_source 测试。"""

    def test_use_f0_and_protect_returns_clone(self):
        """use_f0=1 且 protect>0 时返回克隆。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=0.5)
        self.assertIsNotNone(result)
        self.assertIsNot(result, feats)
        self.assertTrue(torch.equal(result, feats))

    def test_use_f0_zero_returns_none(self):
        """use_f0=0 时返回 None。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=0, protect=0.5)
        self.assertIsNone(result)

    def test_protect_zero_returns_none(self):
        """protect=0 时返回 None。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=0.0)
        self.assertIsNone(result)

    def test_protect_negative_returns_none(self):
        """protect<0 时返回 None。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=-0.1)
        self.assertIsNone(result)

    def test_clone_independent(self):
        """克隆是独立的，修改原张量不影响克隆。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=0.5)
        feats[0, 0, 0] = 999.0
        self.assertNotEqual(result[0, 0, 0].item(), 999.0)

    def test_various_protect_values(self):
        feats = torch.randn(1, 10, 768)
        for protect in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]:
            result = clone_protect_source(feats, use_f0=1, protect=protect)
            self.assertIsNotNone(result)
            self.assertTrue(torch.equal(result, feats))

    def test_various_shapes(self):
        for shape in [(1, 5, 768), (1, 10, 256), (1, 20, 768), (1, 100, 768)]:
            feats = torch.randn(*shape)
            result = clone_protect_source(feats, use_f0=1, protect=0.5)
            self.assertEqual(result.shape, shape)


class TestProtectBlend(unittest.TestCase):
    """protect_blend 测试（硬阈值模式）。"""

    def setUp(self):
        # mock experimental_config 使用硬阈值模式
        self.experimental_patcher = patch("rvc.core.experimental.experimental_config")
        self.mock_config = self.experimental_patcher.start()
        self.mock_config.protect_soft_enabled = False
        self.addCleanup(self.experimental_patcher.stop)

    def test_voiced_full_conversion(self):
        """浊音（pitchf > 0）→ 全转换，不混合原特征。"""
        feats_converted = torch.ones(1, 10, 768)
        feats_original = torch.zeros(1, 10, 768)
        pitchf = torch.ones(1, 10) * 200  # 浊音
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        self.assertTrue(torch.allclose(result, feats_converted))

    def test_unvoiced_mixed(self):
        """清音（pitchf = 0）→ 按 protect 混合原特征。"""
        feats_converted = torch.ones(1, 10, 768)
        feats_original = torch.zeros(1, 10, 768)
        pitchf = torch.zeros(1, 10)  # 清音
        protect = 0.5
        result = protect_blend(feats_converted, feats_original, pitchf, protect=protect)
        # mix = 1.0 - protect = 0.5
        # result = converted * 0.5 + original * 0.5 = 0.5
        expected = torch.ones(1, 10, 768) * 0.5
        self.assertTrue(torch.allclose(result, expected))

    def test_protect_zero_no_mix(self):
        """protect=0 时不混合，全转换。"""
        feats_converted = torch.ones(1, 10, 768)
        feats_original = torch.zeros(1, 10, 768)
        pitchf = torch.zeros(1, 10)
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.0)
        self.assertTrue(torch.allclose(result, feats_converted))

    def test_protect_one_full_original_for_unvoiced(self):
        """protect=1 时清音全用原特征。"""
        feats_converted = torch.ones(1, 10, 768)
        feats_original = torch.zeros(1, 10, 768)
        pitchf = torch.zeros(1, 10)
        result = protect_blend(feats_converted, feats_original, pitchf, protect=1.0)
        self.assertTrue(torch.allclose(result, feats_original))

    def test_mixed_voiced_unvoiced(self):
        """部分浊音部分清音。"""
        feats_converted = torch.ones(1, 4, 768)
        feats_original = torch.zeros(1, 4, 768)
        pitchf = torch.tensor([[200.0, 0.0, 150.0, 0.0]])
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        # 帧 0, 2 是浊音 → 全转换 (1.0)
        self.assertAlmostEqual(result[0, 0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(result[0, 2, 0].item(), 1.0, places=4)
        # 帧 1, 3 是清音 → 混合 (0.5)
        self.assertAlmostEqual(result[0, 1, 0].item(), 0.5, places=4)
        self.assertAlmostEqual(result[0, 3, 0].item(), 0.5, places=4)

    def test_output_shape(self):
        feats_converted = torch.randn(1, 10, 768)
        feats_original = torch.randn(1, 10, 768)
        pitchf = torch.randn(1, 10) * 100
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        self.assertEqual(result.shape, (1, 10, 768))

    def test_various_protect_values(self):
        feats_converted = torch.ones(1, 5, 768)
        feats_original = torch.zeros(1, 5, 768)
        pitchf = torch.zeros(1, 5)
        for protect in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = protect_blend(feats_converted, feats_original, pitchf, protect=protect)
            expected = 1.0 - protect
            self.assertAlmostEqual(result[0, 0, 0].item(), expected, places=4)


class TestProtectBlendSoft(unittest.TestCase):
    """protect_blend 测试（软阈值模式）。"""

    def setUp(self):
        self.experimental_patcher = patch("rvc.core.experimental.experimental_config")
        self.mock_config = self.experimental_patcher.start()
        self.mock_config.protect_soft_enabled = True
        self.mock_config.protect_soft_threshold_hz = 50.0
        self.mock_config.protect_soft_width = 20.0
        self.addCleanup(self.experimental_patcher.stop)

    def test_high_pitch_full_conversion(self):
        """高音高 → 全转换。"""
        feats_converted = torch.ones(1, 5, 768)
        feats_original = torch.zeros(1, 5, 768)
        pitchf = torch.ones(1, 5) * 500  # 远高于阈值
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        # uv_prob 接近 0，mix 接近 1，结果接近全转换
        self.assertGreater(result[0, 0, 0].item(), 0.9)

    def test_zero_pitch_mixed(self):
        """零音高 → 按 protect 混合。"""
        feats_converted = torch.ones(1, 5, 768)
        feats_original = torch.zeros(1, 5, 768)
        pitchf = torch.zeros(1, 5)
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        # uv_prob 接近 1，mix 接近 0.5
        self.assertLess(result[0, 0, 0].item(), 0.6)
        self.assertGreater(result[0, 0, 0].item(), 0.4)

    def test_output_shape(self):
        feats_converted = torch.randn(1, 10, 768)
        feats_original = torch.randn(1, 10, 768)
        pitchf = torch.randn(1, 10).abs() * 100
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        self.assertEqual(result.shape, (1, 10, 768))

    def test_smooth_transition(self):
        """软阈值模式下过渡是平滑的（相邻帧差异不大）。"""
        feats_converted = torch.ones(1, 10, 768)
        feats_original = torch.zeros(1, 10, 768)
        pitchf = torch.linspace(0, 100, 10).unsqueeze(0)
        result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        # 相邻帧的差异应该小于硬阈值模式
        diffs = torch.abs(torch.diff(result[0, :, 0]))
        self.assertTrue(torch.all(diffs < 0.5))


class TestUpsampleFeatures(unittest.TestCase):
    """upsample_features 测试。"""

    def test_basic_upsample(self):
        """基本上采样：scale_factor=2。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False)
        # 上采样后长度翻倍，然后截断到 p_len
        self.assertEqual(result.shape, (1, 20, 768))

    def test_p_len_truncation(self):
        """p_len 小于上采样后长度时截断。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=15, is_half=False)
        self.assertEqual(result.shape, (1, 15, 768))

    def test_p_len_larger_than_input(self):
        """p_len 大于上采样后长度时不填充（只截断）。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=25, is_half=False)
        # 上采样后长度为 20，p_len=25 但只截断不填充，所以长度为 20
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[2], 768)
        self.assertLessEqual(result.shape[1], 25)

    def test_is_half_true(self):
        """is_half=True 且有 feats0/pitchf 时输出 half。"""
        feats = torch.randn(1, 10, 768)
        feats0 = torch.randn(1, 10, 768)
        pitchf = torch.ones(1, 20) * 200
        with patch("rvc.core.experimental.experimental_config") as mock_config:
            mock_config.protect_soft_enabled = False
            result = upsample_features(feats, p_len=20, is_half=True,
                                        feats0=feats0, pitchf=pitchf, protect=0.5)
        self.assertEqual(result.dtype, torch.float16)

    def test_is_half_false_no_feats0(self):
        """没有 feats0 时 is_half 不影响输出 dtype。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=True)
        self.assertEqual(result.dtype, torch.float32)

    def test_with_feats0_and_pitchf(self):
        """有 feats0 和 pitchf 时进行保护混合。"""
        feats = torch.ones(1, 10, 768)
        feats0 = torch.zeros(1, 10, 768)
        pitchf = torch.ones(1, 20) * 200  # 全浊音
        with patch("rvc.core.experimental.experimental_config") as mock_config:
            mock_config.protect_soft_enabled = False
            result = upsample_features(feats, p_len=20, is_half=False,
                                        feats0=feats0, pitchf=pitchf, protect=0.5)
        # 全浊音 → 全转换，结果应该全 1
        self.assertTrue(torch.allclose(result, torch.ones_like(result)))

    def test_feats0_none_skips_blend(self):
        """feats0=None 时不进行保护混合。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False,
                                    feats0=None, pitchf=None, protect=0.5)
        self.assertEqual(result.shape, (1, 20, 768))

    def test_pitchf_none_skips_blend(self):
        """pitchf=None 时不进行保护混合。"""
        feats = torch.randn(1, 10, 768)
        feats0 = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False,
                                    feats0=feats0, pitchf=None, protect=0.5)
        self.assertEqual(result.shape, (1, 20, 768))

    def test_various_feature_dims(self):
        for dim in [256, 512, 768, 1024]:
            feats = torch.randn(1, 10, dim)
            result = upsample_features(feats, p_len=20, is_half=False)
            self.assertEqual(result.shape, (1, 20, dim))

    def test_various_input_lengths(self):
        for length in [5, 10, 20, 50]:
            feats = torch.randn(1, length, 768)
            result = upsample_features(feats, p_len=length * 2, is_half=False)
            self.assertEqual(result.shape[1], length * 2)


class TestFeatureProcessingEdgeCases(unittest.TestCase):
    """特征处理边界条件测试。"""

    def test_clone_protect_source_protect_exactly_zero(self):
        """protect 恰好为 0 时返回 None。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=0.0)
        self.assertIsNone(result)

    def test_clone_protect_source_use_f0_exactly_one(self):
        """use_f0 恰好为 1 且 protect>0 时返回克隆。"""
        feats = torch.randn(1, 10, 768)
        result = clone_protect_source(feats, use_f0=1, protect=0.001)
        self.assertIsNotNone(result)

    def test_protect_blend_protect_negative(self):
        """protect 为负时行为（mix = 1.0 - (-0.5) = 1.5，可能超范围）。"""
        feats_converted = torch.ones(1, 5, 768)
        feats_original = torch.zeros(1, 5, 768)
        pitchf = torch.zeros(1, 5)
        with patch("rvc.core.experimental.experimental_config") as mock_config:
            mock_config.protect_soft_enabled = False
            result = protect_blend(feats_converted, feats_original, pitchf, protect=-0.5)
        # mix = 1.0 - (-0.5) = 1.5，结果 = 1.5
        self.assertAlmostEqual(result[0, 0, 0].item(), 1.5, places=4)

    def test_upsample_p_len_zero(self):
        """p_len=0 时返回空特征。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=0, is_half=False)
        self.assertEqual(result.shape[1], 0)

    def test_upsample_single_frame(self):
        """单帧输入上采样。"""
        feats = torch.randn(1, 1, 768)
        result = upsample_features(feats, p_len=2, is_half=False)
        self.assertEqual(result.shape, (1, 2, 768))

    def test_upsample_large_p_len(self):
        """大 p_len 不崩溃。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=1000, is_half=False)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[2], 768)

    def test_protect_blend_all_unvoiced_protect_half(self):
        """全清音 protect=0.5 时结果为 0.5。"""
        feats_converted = torch.ones(1, 10, 256)
        feats_original = torch.zeros(1, 10, 256)
        pitchf = torch.zeros(1, 10)
        with patch("rvc.core.experimental.experimental_config") as mock_config:
            mock_config.protect_soft_enabled = False
            result = protect_blend(feats_converted, feats_original, pitchf, protect=0.5)
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.5)))

    def test_clone_preserves_dtype(self):
        """克隆保留 dtype。"""
        feats = torch.randn(1, 10, 768).half()
        result = clone_protect_source(feats, use_f0=1, protect=0.5)
        self.assertEqual(result.dtype, torch.float16)

    def test_upsample_output_dtype_float32_default(self):
        """默认输出 float32。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False)
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
