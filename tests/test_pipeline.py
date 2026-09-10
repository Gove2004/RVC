"""InferencePipeline 单元测试 — 状态管理 / 音高缓存 / 参数计算 / 无状态设计。

不依赖模型加载（load() 需要真实 pth 文件），只测试：
- 初始化和状态管理
- reset_pitch_cache
- 无状态设计验证（pipeline 不持有推理参数）
- 辅助方法的参数计算逻辑
"""
import unittest
from types import SimpleNamespace

import torch

from rvc.core.config import InferenceConfig
from rvc.inference.pipeline import InferencePipeline


class TestInferencePipelineInit(unittest.TestCase):
    """InferencePipeline 初始化测试。"""

    def _make_pipeline(self, device="cpu", is_half=False, hubert="base"):
        device_config = SimpleNamespace(device=device, is_half=is_half)
        return InferencePipeline(device_config, "test.pth", hubert=hubert)

    def test_init_sets_basic_attrs(self):
        """初始化时设置基本属性。"""
        pipeline = self._make_pipeline(device="cuda", is_half=True, hubert="chinese")
        self.assertEqual(pipeline.device, "cuda")
        self.assertTrue(pipeline.is_half)
        self.assertEqual(pipeline.pth_path, "test.pth")
        self.assertEqual(pipeline.hubert_variant, "chinese")

    def test_init_model_refs_none(self):
        """初始化时模型引用为 None（load 后才填充）。"""
        pipeline = self._make_pipeline()
        self.assertIsNone(pipeline.hubert_model)
        self.assertIsNone(pipeline.synthesizer)
        self.assertIsNone(pipeline.target_sr)
        self.assertEqual(pipeline.use_f0, 1)

    def test_init_creates_pitch_cache(self):
        """初始化时创建音高缓存。"""
        pipeline = self._make_pipeline()
        self.assertIsNotNone(pipeline.pitch_cache)
        self.assertIsNotNone(pipeline.pitchf_cache)
        # 音高缓存初始全零
        self.assertTrue(torch.all(pipeline.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.pitchf_cache == 0))

    def test_init_creates_cache_dicts(self):
        """初始化时创建缓存字典。"""
        pipeline = self._make_pipeline()
        self.assertIsInstance(pipeline.resample_kernel, dict)
        self.assertIsInstance(pipeline._long_tensor_cache, dict)
        self.assertEqual(len(pipeline.resample_kernel), 0)
        self.assertEqual(len(pipeline._long_tensor_cache), 0)

    def test_init_uses_default_cache(self):
        """不传入 inference_cache 时使用 default_inference_cache。"""
        from rvc.inference.inference_cache import default_inference_cache
        pipeline = self._make_pipeline()
        self.assertIs(pipeline.inference_cache, default_inference_cache)

    def test_init_uses_custom_cache(self):
        """传入自定义 inference_cache 时使用自定义缓存。"""
        from rvc.inference.inference_cache import InferenceCache
        custom_cache = InferenceCache()
        device_config = SimpleNamespace(device="cpu", is_half=False)
        pipeline = InferencePipeline(device_config, "test.pth", inference_cache=custom_cache)
        self.assertIs(pipeline.inference_cache, custom_cache)


class TestInferencePipelinePitchCache(unittest.TestCase):
    """音高缓存管理测试。"""

    def setUp(self):
        device_config = SimpleNamespace(device="cpu", is_half=False)
        self.pipeline = InferencePipeline(device_config, "test.pth")

    def test_reset_pitch_cache_clears(self):
        """reset_pitch_cache 清空音高缓存。"""
        # 先修改缓存（模拟推理后状态）
        self.pipeline.pitch_cache.fill_(1.0)
        self.pipeline.pitchf_cache.fill_(2.0)
        self.assertFalse(torch.all(self.pipeline.pitch_cache == 0))
        self.assertFalse(torch.all(self.pipeline.pitchf_cache == 0))

        self.pipeline.reset_pitch_cache()

        self.assertTrue(torch.all(self.pipeline.pitch_cache == 0))
        self.assertTrue(torch.all(self.pipeline.pitchf_cache == 0))

    def test_pitch_cache_device(self):
        """音高缓存在正确的设备上。"""
        self.assertEqual(self.pipeline.pitch_cache.device.type, "cpu")
        self.assertEqual(self.pipeline.pitchf_cache.device.type, "cpu")

    def test_pitch_cache_shape(self):
        """音高缓存形状正确。"""
        # pitch_cache 形状应为 (cache_size,)，pitchf_cache 类似
        self.assertEqual(self.pipeline.pitch_cache.dim(), 1)
        self.assertEqual(self.pipeline.pitchf_cache.dim(), 1)
        self.assertGreater(self.pipeline.pitch_cache.shape[0], 0)
        self.assertGreater(self.pipeline.pitchf_cache.shape[0], 0)


class TestInferencePipelineStatelessDesign(unittest.TestCase):
    """无状态设计验证 — pipeline 不持有推理参数，每次 infer 从 config 传入。"""

    def setUp(self):
        device_config = SimpleNamespace(device="cpu", is_half=False)
        self.pipeline = InferencePipeline(device_config, "test.pth")

    def test_no_inference_config_attr(self):
        """pipeline 不持有 InferenceConfig 实例。"""
        self.assertFalse(hasattr(self.pipeline, "inference_config"))
        self.assertFalse(hasattr(self.pipeline, "config"))

    def test_no_runtime_param_attrs(self):
        """pipeline 不持有运行时参数（pitch/formant/protect 等）。"""
        for attr in ("pitch", "formant", "protect", "rms_mix", "f0_method"):
            self.assertFalse(hasattr(self.pipeline, attr),
                             f"pipeline 不应持有 {attr}（无状态设计）")

    def test_infer_requires_config_param(self):
        """infer 方法签名要求传入 config 参数。"""
        import inspect
        sig = inspect.signature(self.pipeline.infer)
        self.assertIn("config", sig.parameters)
        self.assertEqual(sig.parameters["config"].annotation, InferenceConfig)


class TestInferenceConfigDefaults(unittest.TestCase):
    """InferenceConfig 默认值测试（pipeline 推理时使用）。"""

    def test_default_pitch(self):
        config = InferenceConfig()
        self.assertEqual(config.pitch, 0)

    def test_default_formant(self):
        config = InferenceConfig()
        self.assertEqual(config.formant, 0.0)

    def test_default_protect(self):
        config = InferenceConfig()
        self.assertEqual(config.protect, 0.5)

    def test_default_rms_mix(self):
        config = InferenceConfig()
        self.assertEqual(config.rms_mix, 0.0)

    def test_default_f0_method(self):
        config = InferenceConfig()
        self.assertEqual(config.f0_method, "rmvpe")


class TestInferencePipelineFormantCalculation(unittest.TestCase):
    """共振峰参数计算逻辑测试（从 _infer_impl 提取的纯计算逻辑）。"""

    def test_formant_factor_zero(self):
        """formant=0 时 factor=1.0。"""
        formant = 0.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 1.0)

    def test_formant_factor_positive(self):
        """formant=12 时 factor=2.0（升高一个八度）。"""
        formant = 12.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 2.0)

    def test_formant_factor_negative(self):
        """formant=-12 时 factor=0.5（降低一个八度）。"""
        formant = -12.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 0.5)

    def test_return_length2_proportional_to_factor(self):
        """return_length2 与 factor 成正比。"""
        import numpy as np
        return_length = 100
        for formant in [0, 6, 12, -6, -12]:
            factor = pow(2, formant / 12)
            return_length2 = int(np.ceil(return_length * factor))
            if formant == 0:
                self.assertEqual(return_length2, return_length)
            elif formant > 0:
                self.assertGreater(return_length2, return_length)
            else:
                self.assertLess(return_length2, return_length)


if __name__ == "__main__":
    unittest.main()
