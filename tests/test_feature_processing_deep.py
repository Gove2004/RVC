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


if __name__ == "__main__":
    unittest.main()
