"""推理管道深度测试 — InferencePipeline 初始化/缓存状态/重置（不需要真实模型）。

覆盖：
- InferencePipeline 初始化: device/is_half/inference_cache/pth_path/hubert_variant
- 缓存状态: pitch_cache/pitchf_cache/resample_kernel/long_tensor_cache
- 模型引用初始为 None
- reset_pitch_cache: 重置音高缓存为零
- 注意：load/infer 需要真实模型，只测试不调用真实模型的部分
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rvc.inference.pipeline import InferencePipeline


class TestInferencePipelineInit(unittest.TestCase):
    """InferencePipeline 初始化测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_creates_pipeline(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline)

    def test_device_stored(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.device, "cpu")

    def test_is_half_stored(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertFalse(pipeline.is_half)

    def test_pth_path_stored(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.pth_path, "/path/model.pth")

    def test_default_hubert_variant(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.hubert_variant, "chinese")

    def test_custom_hubert_variant(self):
        pipeline = InferencePipeline(
            self.device_config, "/path/model.pth", hubert="base",
        )
        self.assertEqual(pipeline.hubert_variant, "base")

    def test_cuda_device(self):
        device_config = SimpleNamespace(device="cuda:0", is_half=True)
        pipeline = InferencePipeline(device_config, "/path/model.pth")
        self.assertEqual(pipeline.device, "cuda:0")
        self.assertTrue(pipeline.is_half)

    def test_various_pth_paths(self):
        for path in ["/a.pth", "/b/c.pth", "C:\\Models\\model.pth", "relative.pth"]:
            pipeline = InferencePipeline(self.device_config, path)
            self.assertEqual(pipeline.pth_path, path)


class TestInferencePipelineCacheState(unittest.TestCase):
    """缓存状态初始化测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_pitch_cache_created(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline.pitch_cache)
        self.assertIsInstance(pipeline.pitch_cache, torch.Tensor)

    def test_pitchf_cache_created(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline.pitchf_cache)
        self.assertIsInstance(pipeline.pitchf_cache, torch.Tensor)

    def test_pitch_cache_on_correct_device(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(str(pipeline.pitch_cache.device), "cpu")

    def test_resample_kernel_empty_dict(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsInstance(pipeline.resample_kernel, dict)
        self.assertEqual(len(pipeline.resample_kernel), 0)

    def test_long_tensor_cache_empty_dict(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsInstance(pipeline._long_tensor_cache, dict)
        self.assertEqual(len(pipeline._long_tensor_cache), 0)

    def test_pitch_cache_shape(self):
        """pitch_cache 应该是 1D 张量（用于滚动缓存）。"""
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.pitch_cache.dim(), 1)
        self.assertGreater(pipeline.pitch_cache.shape[0], 0)

    def test_pitchf_cache_shape(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.pitchf_cache.dim(), 1)
        self.assertGreater(pipeline.pitchf_cache.shape[0], 0)

    def test_pitch_cache_dtype_long(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.pitch_cache.dtype, torch.long)

    def test_pitchf_cache_dtype_float(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.pitchf_cache.dtype, torch.float32)


class TestInferencePipelineModelRefs(unittest.TestCase):
    """模型引用初始状态测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_hubert_model_none_initially(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.hubert_model)

    def test_synthesizer_none_initially(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.synthesizer)

    def test_target_sr_none_initially(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.target_sr)

    def test_use_f0_default_1(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.use_f0, 1)


class TestInferencePipelineResetPitchCache(unittest.TestCase):
    """reset_pitch_cache 测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_reset_sets_pitch_cache_to_zero(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        # 先填充非零值
        pipeline.pitch_cache.fill_(5)
        pipeline.pitchf_cache.fill_(3.14)
        # 重置
        pipeline.reset_pitch_cache()
        # 验证为零
        self.assertTrue(torch.all(pipeline.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.pitchf_cache == 0.0))

    def test_reset_preserves_shape(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        original_shape = pipeline.pitch_cache.shape
        original_f_shape = pipeline.pitchf_cache.shape
        pipeline.reset_pitch_cache()
        self.assertEqual(pipeline.pitch_cache.shape, original_shape)
        self.assertEqual(pipeline.pitchf_cache.shape, original_f_shape)

    def test_reset_multiple_times(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        for _ in range(5):
            pipeline.pitch_cache.fill_(99)
            pipeline.reset_pitch_cache()
            self.assertTrue(torch.all(pipeline.pitch_cache == 0))

    def test_reset_after_partial_fill(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        pipeline.pitch_cache[:10] = 42
        pipeline.pitchf_cache[:10] = 1.5
        pipeline.reset_pitch_cache()
        self.assertTrue(torch.all(pipeline.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.pitchf_cache == 0.0))


class TestInferencePipelineInferenceCache(unittest.TestCase):
    """inference_cache 参数测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_default_inference_cache(self):
        """inference_cache=None 时使用默认全局缓存。"""
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline.inference_cache)

    def test_custom_inference_cache(self):
        custom_cache = MagicMock()
        pipeline = InferencePipeline(
            self.device_config, "/path/model.pth", inference_cache=custom_cache,
        )
        self.assertIs(pipeline.inference_cache, custom_cache)

    def test_custom_cache_used_in_init(self):
        """自定义缓存在初始化时被使用（pitch_cache 创建不依赖它）。"""
        custom_cache = MagicMock()
        pipeline = InferencePipeline(
            self.device_config, "/path/model.pth", inference_cache=custom_cache,
        )
        # pitch_cache 应该正常创建
        self.assertIsNotNone(pipeline.pitch_cache)


class TestInferencePipelineAttributes(unittest.TestCase):
    """其他属性测试。"""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_has_load_method(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertTrue(callable(pipeline.load))

    def test_has_infer_method(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertTrue(callable(pipeline.infer))

    def test_has_reset_pitch_cache_method(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertTrue(callable(pipeline.reset_pitch_cache))

    def test_independent_instances(self):
        """不同 pipeline 实例不共享缓存状态。"""
        p1 = InferencePipeline(self.device_config, "/path/model1.pth")
        p2 = InferencePipeline(self.device_config, "/path/model2.pth")
        p1.pitch_cache.fill_(5)
        self.assertTrue(torch.all(p2.pitch_cache == 0))

    def test_independent_resample_kernels(self):
        p1 = InferencePipeline(self.device_config, "/path/model1.pth")
        p2 = InferencePipeline(self.device_config, "/path/model2.pth")
        p1.resample_kernel["key"] = "value"
        self.assertNotIn("key", p2.resample_kernel)


if __name__ == "__main__":
    unittest.main()
