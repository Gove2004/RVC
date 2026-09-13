"""特征处理深度测试 — upsample_features + 清辅音保护。

覆盖：
- upsample_features: 各种输入形状/p_len/is_half
- 清辅音保护: F0=0 帧混合原始特征，protect=0.5 关闭
- 注意：extract_hubert_features 需要真实模型，只测试不调用真实模型的部分
"""
import unittest

import torch

from rvc.inference.feature_processing import upsample_features


class TestUpsampleFeatures(unittest.TestCase):
    """upsample_features 测试（简化后 3 参数版本）。"""

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
        """is_half=True 时输出 half。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=True)
        self.assertEqual(result.dtype, torch.float16)

    def test_is_half_false(self):
        """is_half=False 时输出 float32。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False)
        self.assertEqual(result.dtype, torch.float32)

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

    def test_batch_dimension_preserved(self):
        """batch 维度保持不变。"""
        for batch in [1, 2, 4]:
            feats = torch.randn(batch, 10, 768)
            result = upsample_features(feats, p_len=20, is_half=False)
            self.assertEqual(result.shape[0], batch)


class TestUpsampleFeaturesEdgeCases(unittest.TestCase):
    """upsample_features 边界条件测试。"""

    def test_p_len_zero(self):
        """p_len=0 时返回空特征。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=0, is_half=False)
        self.assertEqual(result.shape[1], 0)

    def test_single_frame(self):
        """单帧输入上采样。"""
        feats = torch.randn(1, 1, 768)
        result = upsample_features(feats, p_len=2, is_half=False)
        self.assertEqual(result.shape, (1, 2, 768))

    def test_large_p_len(self):
        """大 p_len 不崩溃。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=1000, is_half=False)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[2], 768)

    def test_output_dtype_float32_default(self):
        """默认输出 float32。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=False)
        self.assertEqual(result.dtype, torch.float32)

    def test_half_output_dtype(self):
        """is_half=True 输出 float16。"""
        feats = torch.randn(1, 10, 768)
        result = upsample_features(feats, p_len=20, is_half=True)
        self.assertEqual(result.dtype, torch.float16)


class TestConsonantProtection(unittest.TestCase):
    """清辅音保护测试 — F0=0 帧混合原始特征。"""

    def test_protect_disabled_at_0_5(self):
        """protect=0.5 时关闭保护，输出与不保护一致。"""
        feats = torch.randn(1, 10, 768)
        feats0 = torch.randn(1, 10, 768)
        pitchf = torch.tensor([[0.0, 100.0, 0.0, 200.0, 0.0, 150.0, 0.0, 180.0, 0.0, 120.0]])
        result_with = upsample_features(feats, 20, False, feats0=feats0, pitchf=pitchf, protect=0.5)
        result_without = upsample_features(feats, 20, False)
        self.assertTrue(torch.allclose(result_with, result_without, atol=1e-6))

    def test_unvoiced_frames_mixed(self):
        """F0=0 的清音帧混合原始特征。"""
        feats = torch.ones(1, 5, 768) * 0.5  # 合成特征全 0.5
        feats0 = torch.ones(1, 5, 768) * 1.0  # 原始特征全 1.0
        # pitchf 长度 = p_len = 10（100fps，与上采样后特征对齐）
        pitchf = torch.tensor([[0.0, 100.0, 0.0, 200.0, 150.0, 0.0, 100.0, 0.0, 200.0, 150.0]])
        result = upsample_features(feats, 10, False, feats0=feats0, pitchf=pitchf, protect=0.0)
        # protect=0.0 → strength=1.0，清音帧完全用原始特征
        self.assertAlmostEqual(result[0, 0, 0].item(), 1.0, places=4)  # 清音
        self.assertAlmostEqual(result[0, 1, 0].item(), 0.5, places=4)  # 浊音
        self.assertAlmostEqual(result[0, 2, 0].item(), 1.0, places=4)  # 清音
        self.assertAlmostEqual(result[0, 3, 0].item(), 0.5, places=4)  # 浊音

    def test_voiced_frames_unchanged(self):
        """浊音帧（F0!=0）不受保护影响。"""
        feats = torch.randn(1, 5, 768)
        feats0 = torch.randn(1, 5, 768)
        pitchf = torch.ones(1, 10) * 100.0  # 全浊音，长度=p_len
        result = upsample_features(feats, 10, False, feats0=feats0, pitchf=pitchf, protect=0.0)
        result_no_protect = upsample_features(feats, 10, False)
        self.assertTrue(torch.allclose(result, result_no_protect, atol=1e-6))

    def test_partial_protection_strength(self):
        """protect=0.25 时 strength=0.5，清音帧混合一半原始特征。"""
        feats = torch.ones(1, 5, 768) * 0.0  # 合成特征全 0
        feats0 = torch.ones(1, 5, 768) * 1.0  # 原始特征全 1
        pitchf = torch.tensor([[0.0, 100.0, 0.0, 200.0, 150.0, 0.0, 100.0, 0.0, 200.0, 150.0]])
        result = upsample_features(feats, 10, False, feats0=feats0, pitchf=pitchf, protect=0.25)
        # strength = 1 - 0.25/0.5 = 0.5，清音帧 = 0.5*0 + 0.5*1 = 0.5
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.5, places=4)
        # 浊音帧不受影响，还是 0
        self.assertAlmostEqual(result[0, 1, 0].item(), 0.0, places=4)

    def test_no_feats0_no_protection(self):
        """feats0=None 时不启用保护。"""
        feats = torch.randn(1, 5, 768)
        pitchf = torch.tensor([[0.0, 100.0, 0.0, 200.0, 150.0, 0.0, 100.0, 0.0, 200.0, 150.0]])
        result = upsample_features(feats, 10, False, feats0=None, pitchf=pitchf, protect=0.0)
        result_no_protect = upsample_features(feats, 10, False)
        self.assertTrue(torch.allclose(result, result_no_protect, atol=1e-6))

    def test_no_pitchf_no_protection(self):
        """pitchf=None 时不启用保护。"""
        feats = torch.randn(1, 5, 768)
        feats0 = torch.randn(1, 5, 768)
        result = upsample_features(feats, 10, False, feats0=feats0, pitchf=None, protect=0.0)
        result_no_protect = upsample_features(feats, 10, False)
        self.assertTrue(torch.allclose(result, result_no_protect, atol=1e-6))

    def test_output_shape_with_protection(self):
        """启用保护时输出形状正确。"""
        feats = torch.randn(1, 10, 768)
        feats0 = torch.randn(1, 10, 768)
        pitchf = torch.randn(1, 20).abs() * 100  # 长度=p_len=20
        result = upsample_features(feats, 20, False, feats0=feats0, pitchf=pitchf, protect=0.33)
        self.assertEqual(result.shape, (1, 20, 768))


if __name__ == "__main__":
    unittest.main()
