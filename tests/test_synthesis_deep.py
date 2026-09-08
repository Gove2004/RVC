"""Synthesizer 深度测试 — 纯函数部分（缓存张量/音高转换/共振峰重采样）。

覆盖：
- cached_long_tensor: 缓存行为、重复调用返回同一对象、不同值不同对象
- cast_pitch_tensors: is_half True/False、dtype 验证
- apply_formant_resample: factor=1 不处理、factor!=1 重采样、缓存清理
"""
import unittest

import torch

from rvc.inference.synthesis import (
    apply_formant_resample,
    cached_long_tensor,
    cast_pitch_tensors,
)


class TestCachedLongTensor(unittest.TestCase):
    """cached_long_tensor 测试。"""

    def test_creates_tensor(self):
        cache = {}
        tensor = cached_long_tensor(cache, value=5, device="cpu")
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(tensor.item(), 5)

    def test_caches_tensor(self):
        """重复调用同一 value 返回同一对象。"""
        cache = {}
        t1 = cached_long_tensor(cache, value=5, device="cpu")
        t2 = cached_long_tensor(cache, value=5, device="cpu")
        self.assertIs(t1, t2)

    def test_different_values_different_objects(self):
        cache = {}
        t1 = cached_long_tensor(cache, value=5, device="cpu")
        t2 = cached_long_tensor(cache, value=10, device="cpu")
        self.assertIsNot(t1, t2)
        self.assertEqual(t1.item(), 5)
        self.assertEqual(t2.item(), 10)

    def test_cache_populated(self):
        cache = {}
        cached_long_tensor(cache, value=5, device="cpu")
        self.assertIn(5, cache)
        self.assertEqual(cache[5].item(), 5)

    def test_various_values(self):
        cache = {}
        for v in [0, 1, 10, 100, 1000, 255, 109]:
            tensor = cached_long_tensor(cache, value=v, device="cpu")
            self.assertEqual(tensor.item(), v)
            self.assertEqual(tensor.dtype, torch.long)

    def test_value_converted_to_int(self):
        """浮点 value 被转换为 int。"""
        cache = {}
        tensor = cached_long_tensor(cache, value=5.7, device="cpu")
        self.assertEqual(tensor.item(), 5)

    def test_tensor_shape(self):
        cache = {}
        tensor = cached_long_tensor(cache, value=5, device="cpu")
        self.assertEqual(tensor.shape, (1,))

    def test_empty_cache(self):
        cache = {}
        self.assertEqual(len(cache), 0)
        cached_long_tensor(cache, value=5, device="cpu")
        self.assertEqual(len(cache), 1)


class TestCastPitchTensors(unittest.TestCase):
    """cast_pitch_tensors 测试。"""

    def test_is_half_true(self):
        pitch = torch.tensor([1, 2, 3], dtype=torch.long)
        pitchf = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        p, pf = cast_pitch_tensors(pitch, pitchf, is_half=True)
        self.assertEqual(p.dtype, torch.long)
        self.assertEqual(pf.dtype, torch.float16)

    def test_is_half_false(self):
        pitch = torch.tensor([1, 2, 3], dtype=torch.long)
        pitchf = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        p, pf = cast_pitch_tensors(pitch, pitchf, is_half=False)
        self.assertEqual(p.dtype, torch.long)
        self.assertEqual(pf.dtype, torch.float32)

    def test_pitch_always_long(self):
        """pitch 总是 long 类型，无论 is_half。"""
        for is_half in [True, False]:
            pitch = torch.tensor([1.0, 2.0], dtype=torch.float32)
            pitchf = torch.tensor([1.0, 2.0], dtype=torch.float32)
            p, _ = cast_pitch_tensors(pitch, pitchf, is_half=is_half)
            self.assertEqual(p.dtype, torch.long)

    def test_values_preserved(self):
        pitch = torch.tensor([10, 20, 30], dtype=torch.long)
        pitchf = torch.tensor([10.5, 20.5, 30.5], dtype=torch.float32)
        p, pf = cast_pitch_tensors(pitch, pitchf, is_half=False)
        self.assertTrue(torch.equal(p, pitch))
        self.assertTrue(torch.allclose(pf, pitchf))

    def test_various_lengths(self):
        for n in [1, 10, 100, 1000]:
            pitch = torch.randint(1, 256, (n,))
            pitchf = torch.randn(n) * 100 + 200
            p, pf = cast_pitch_tensors(pitch, pitchf, is_half=False)
            self.assertEqual(p.shape, (n,))
            self.assertEqual(pf.shape, (n,))

    def test_half_conversion_accuracy(self):
        """float16 转换保持合理精度。"""
        pitchf = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0], dtype=torch.float32)
        pitch = torch.zeros(5, dtype=torch.long)
        _, pf = cast_pitch_tensors(pitch, pitchf, is_half=True)
        self.assertTrue(torch.allclose(pf.float(), pitchf, rtol=0.01))


class TestApplyFormantResample(unittest.TestCase):
    """apply_formant_resample 测试。"""

    def test_factor_one_returns_original(self):
        """factor=1 时不处理，返回原音频。"""
        audio = torch.randn(1, 1000)
        result = apply_formant_resample(
            audio, factor=1.0, target_sr=16000,
            resample_kernel={}, device="cpu",
        )
        self.assertIs(result, audio)

    def test_factor_near_one_returns_original(self):
        """factor 导致 upp_res == target_sr//100 时不处理。"""
        audio = torch.randn(1, 1000)
        # factor=1.0 -> upp_res = 160, target_sr//100 = 160
        result = apply_formant_resample(
            audio, factor=1.0, target_sr=16000,
            resample_kernel={}, device="cpu",
        )
        self.assertIs(result, audio)

    def test_factor_not_one_creates_kernel(self):
        """factor!=1 时创建重采样核并缓存。"""
        audio = torch.randn(1, 1000)
        kernel = {}
        result = apply_formant_resample(
            audio, factor=1.2, target_sr=16000,
            resample_kernel=kernel, device="cpu",
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(kernel), 1)

    def test_kernel_cached(self):
        """重复调用同一 factor 使用缓存的核。"""
        audio = torch.randn(1, 1000)
        kernel = {}
        result1 = apply_formant_resample(
            audio, factor=1.2, target_sr=16000,
            resample_kernel=kernel, device="cpu",
        )
        kernel_copy = dict(kernel)
        result2 = apply_formant_resample(
            audio, factor=1.2, target_sr=16000,
            resample_kernel=kernel, device="cpu",
        )
        # 第二次调用不应该创建新的核
        self.assertEqual(len(kernel), len(kernel_copy))

    def test_kernel_cache_limit_64(self):
        """缓存超过 64 个时清空。"""
        audio = torch.randn(1, 100)
        kernel = {}
        # 填充 64 个不同的 factor
        for i in range(64):
            factor = 1.0 + (i + 1) * 0.01
            apply_formant_resample(
                audio, factor=factor, target_sr=16000,
                resample_kernel=kernel, device="cpu",
            )
        self.assertEqual(len(kernel), 64)
        # 第 65 个应该触发清空
        apply_formant_resample(
            audio, factor=2.0, target_sr=16000,
            resample_kernel=kernel, device="cpu",
        )
        self.assertEqual(len(kernel), 1)

    def test_output_is_tensor(self):
        audio = torch.randn(1, 1000)
        result = apply_formant_resample(
            audio, factor=1.2, target_sr=16000,
            resample_kernel={}, device="cpu",
        )
        self.assertIsInstance(result, torch.Tensor)

    def test_various_factors(self):
        for factor in [0.5, 0.8, 1.1, 1.2, 1.5, 2.0]:
            audio = torch.randn(1, 500)
            result = apply_formant_resample(
                audio, factor=factor, target_sr=16000,
                resample_kernel={}, device="cpu",
            )
            self.assertIsInstance(result, torch.Tensor)

    def test_various_target_sr(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            audio = torch.randn(1, 500)
            result = apply_formant_resample(
                audio, factor=1.2, target_sr=sr,
                resample_kernel={}, device="cpu",
            )
            self.assertIsInstance(result, torch.Tensor)

    def test_batch_dimension_preserved(self):
        """批量维度保持不变。"""
        audio = torch.randn(4, 1000)
        result = apply_formant_resample(
            audio, factor=1.2, target_sr=16000,
            resample_kernel={}, device="cpu",
        )
        self.assertEqual(result.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
