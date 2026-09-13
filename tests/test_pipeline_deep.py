"""InferencePipeline deep tests - initialization, cache state, reset (no real model needed)."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from rvc.inference.pipeline import InferencePipeline


class TestInferencePipelineInit(unittest.TestCase):
    """Tests for InferencePipeline initialization."""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_init_creates_cache_dicts(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsInstance(pipeline.state.resample_kernel, dict)
        self.assertIsInstance(pipeline.state.long_tensor_cache, dict)

    def test_init_creates_pitch_cache(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline.state.pitch_cache)
        self.assertIsInstance(pipeline.state.pitch_cache, torch.Tensor)

    def test_pitch_cache_device(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.state.pitch_cache.device.type, "cpu")

    def test_pitch_cache_shape(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.state.pitch_cache.dim(), 1)
        self.assertGreater(pipeline.state.pitch_cache.shape[0], 0)

    def test_pitchf_cache_shape(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.state.pitchf_cache.dim(), 1)
        self.assertGreater(pipeline.state.pitchf_cache.shape[0], 0)

    def test_pitch_cache_dtype_long(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.state.pitch_cache.dtype, torch.long)

    def test_pitchf_cache_dtype_float(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertEqual(pipeline.state.pitchf_cache.dtype, torch.float32)

    def test_reset_pitch_cache_clears(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        pipeline.state.pitch_cache.fill_(5)
        pipeline.reset_pitch_cache()
        self.assertTrue(torch.all(pipeline.state.pitch_cache == 0))


class TestInferencePipelineModelRefs(unittest.TestCase):
    """Tests for model reference initialization state."""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_target_sr_none_initially(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.state.target_sr)

    def test_hubert_model_none(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.state.hubert_model)

    def test_synthesizer_none(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNone(pipeline.state.synthesizer)


class TestInferencePipelineResetPitchCache(unittest.TestCase):
    """Tests for reset_pitch_cache."""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_reset_sets_pitch_cache_to_zero(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        pipeline.state.pitch_cache.fill_(5)
        pipeline.state.pitchf_cache.fill_(3.14)
        pipeline.reset_pitch_cache()
        self.assertTrue(torch.all(pipeline.state.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.state.pitchf_cache == 0.0))

    def test_reset_preserves_shape(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        original_shape = pipeline.state.pitch_cache.shape
        original_f_shape = pipeline.state.pitchf_cache.shape
        pipeline.reset_pitch_cache()
        self.assertEqual(pipeline.state.pitch_cache.shape, original_shape)
        self.assertEqual(pipeline.state.pitchf_cache.shape, original_f_shape)

    def test_reset_multiple_times(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        for _ in range(5):
            pipeline.state.pitch_cache.fill_(99)
            pipeline.reset_pitch_cache()
            self.assertTrue(torch.all(pipeline.state.pitch_cache == 0))

    def test_reset_after_partial_fill(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        pipeline.state.pitch_cache[:10] = 42
        pipeline.state.pitchf_cache[:10] = 1.5
        pipeline.reset_pitch_cache()
        self.assertTrue(torch.all(pipeline.state.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.state.pitchf_cache == 0.0))


class TestInferencePipelineInferenceCache(unittest.TestCase):
    """Tests for inference_cache parameter."""

    def setUp(self):
        self.device_config = SimpleNamespace(device="cpu", is_half=False)

    def test_default_inference_cache(self):
        pipeline = InferencePipeline(self.device_config, "/path/model.pth")
        self.assertIsNotNone(pipeline.state.inference_cache)

    def test_custom_inference_cache(self):
        custom_cache = MagicMock()
        pipeline = InferencePipeline(
            self.device_config, "/path/model.pth", inference_cache=custom_cache,
        )
        self.assertIs(pipeline.state.inference_cache, custom_cache)


class TestInferencePipelineAttributes(unittest.TestCase):
    """Tests for other pipeline attributes."""

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
        p1 = InferencePipeline(self.device_config, "/path/model1.pth")
        p2 = InferencePipeline(self.device_config, "/path/model2.pth")
        p1.state.pitch_cache.fill_(5)
        self.assertTrue(torch.all(p2.state.pitch_cache == 0))

    def test_independent_resample_kernels(self):
        p1 = InferencePipeline(self.device_config, "/path/model1.pth")
        p2 = InferencePipeline(self.device_config, "/path/model2.pth")
        p1.state.resample_kernel["key"] = "value"
        self.assertNotIn("key", p2.state.resample_kernel)


if __name__ == "__main__":
    unittest.main()
