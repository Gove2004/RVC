"""InferencePipeline unit tests - state management / pitch cache / stateless design.

Does not depend on model loading (load() requires a real .pth file).
Tests only:
- Initialization and state management
- reset_pitch_cache
- Stateless design verification (pipeline does not hold inference params)
- Helper method parameter calculation logic
"""
import unittest
from types import SimpleNamespace

import torch

from rvc.core.config import InferenceConfig
from rvc.inference.pipeline import InferencePipeline


class TestInferencePipelineInit(unittest.TestCase):
    """Tests for InferencePipeline initialization."""

    def _make_pipeline(self, device="cpu", is_half=False, hubert="base"):
        device_config = SimpleNamespace(device=device, is_half=is_half)
        return InferencePipeline(device_config, "test.pth", hubert=hubert)

    def test_init_sets_basic_attrs(self):
        """Initialization sets basic attributes."""
        pipeline = self._make_pipeline(device="cuda", is_half=True, hubert="chinese")
        self.assertEqual(pipeline.device, "cuda")
        self.assertTrue(pipeline.is_half)
        self.assertEqual(pipeline.pth_path, "test.pth")
        self.assertEqual(pipeline.hubert_variant, "chinese")

    def test_init_model_refs_none(self):
        """Model references are None at init (filled after load)."""
        pipeline = self._make_pipeline()
        self.assertIsNone(pipeline.state.hubert_model)
        self.assertIsNone(pipeline.state.synthesizer)
        self.assertIsNone(pipeline.state.target_sr)
        self.assertEqual(pipeline.use_f0, 1)

    def test_init_creates_pitch_cache(self):
        """Initialization creates pitch cache."""
        pipeline = self._make_pipeline()
        self.assertIsNotNone(pipeline.state.pitch_cache)
        self.assertIsNotNone(pipeline.state.pitchf_cache)
        # Pitch cache initialized to all zeros
        self.assertTrue(torch.all(pipeline.state.pitch_cache == 0))
        self.assertTrue(torch.all(pipeline.state.pitchf_cache == 0))

    def test_init_creates_cache_dicts(self):
        """Initialization creates cache dictionaries."""
        pipeline = self._make_pipeline()
        self.assertIsInstance(pipeline.state.resample_kernel, dict)
        self.assertIsInstance(pipeline.state.long_tensor_cache, dict)
        self.assertEqual(len(pipeline.state.resample_kernel), 0)
        self.assertEqual(len(pipeline.state.long_tensor_cache), 0)

    def test_init_uses_default_cache(self):
        """Uses default_inference_cache when inference_cache is not passed."""
        from rvc.inference.inference_cache import default_inference_cache
        pipeline = self._make_pipeline()
        self.assertIs(pipeline.state.inference_cache, default_inference_cache)

    def test_init_uses_custom_cache(self):
        """Uses custom inference_cache when passed."""
        from rvc.inference.inference_cache import InferenceCache
        custom_cache = InferenceCache()
        device_config = SimpleNamespace(device="cpu", is_half=False)
        pipeline = InferencePipeline(device_config, "test.pth", inference_cache=custom_cache)
        self.assertIs(pipeline.state.inference_cache, custom_cache)


class TestInferencePipelinePitchCache(unittest.TestCase):
    """Tests for pitch cache management."""

    def setUp(self):
        device_config = SimpleNamespace(device="cpu", is_half=False)
        self.pipeline = InferencePipeline(device_config, "test.pth")

    def test_reset_pitch_cache_clears(self):
        """reset_pitch_cache clears pitch cache."""
        # First modify cache (simulate post-inference state)
        self.pipeline.state.pitch_cache.fill_(1.0)
        self.pipeline.state.pitchf_cache.fill_(2.0)
        self.assertFalse(torch.all(self.pipeline.state.pitch_cache == 0))
        self.assertFalse(torch.all(self.pipeline.state.pitchf_cache == 0))

        self.pipeline.reset_pitch_cache()

        self.assertTrue(torch.all(self.pipeline.state.pitch_cache == 0))
        self.assertTrue(torch.all(self.pipeline.state.pitchf_cache == 0))

    def test_pitch_cache_device(self):
        """Pitch cache is on the correct device."""
        self.assertEqual(self.pipeline.state.pitch_cache.device.type, "cpu")
        self.assertEqual(self.pipeline.state.pitchf_cache.device.type, "cpu")

    def test_pitch_cache_shape(self):
        """Pitch cache shape is correct."""
        # pitch_cache shape should be (cache_size,), pitchf_cache similar
        self.assertEqual(self.pipeline.state.pitch_cache.dim(), 1)
        self.assertEqual(self.pipeline.state.pitchf_cache.dim(), 1)
        self.assertGreater(self.pipeline.state.pitch_cache.shape[0], 0)
        self.assertGreater(self.pipeline.state.pitchf_cache.shape[0], 0)


class TestInferencePipelineStatelessDesign(unittest.TestCase):
    """Stateless design verification - pipeline does not hold inference params, each infer takes config."""

    def setUp(self):
        device_config = SimpleNamespace(device="cpu", is_half=False)
        self.pipeline = InferencePipeline(device_config, "test.pth")

    def test_no_inference_config_attr(self):
        """Pipeline does not hold an InferenceConfig instance."""
        self.assertFalse(hasattr(self.pipeline, "inference_config"))
        self.assertFalse(hasattr(self.pipeline, "config"))

    def test_no_runtime_param_attrs(self):
        """Pipeline does not hold runtime params (formant/rms_mix etc.)."""
        for attr in ("formant", "rms_mix", "f0_method"):
            self.assertFalse(hasattr(self.pipeline, attr),
                             f"pipeline should not hold {attr} (stateless design)")

    def test_extract_features_requires_config_param(self):
        """extract_features method signature requires config param."""
        import inspect
        sig = inspect.signature(self.pipeline.extract_features)
        self.assertIn("config", sig.parameters)
        self.assertEqual(sig.parameters["config"].annotation, InferenceConfig)


class TestInferenceConfigDefaults(unittest.TestCase):
    """InferenceConfig default value tests (used during pipeline inference)."""

    def test_default_pitch(self):
        config = InferenceConfig()

    def test_default_formant(self):
        config = InferenceConfig()
        self.assertEqual(config.formant, 0.0)

    def test_default_rms_mix(self):
        config = InferenceConfig()
        self.assertEqual(config.rms_mix, 0.0)

    def test_default_f0_method(self):
        config = InferenceConfig()
        self.assertEqual(config.f0_method, "rmvpe")


class TestInferencePipelineFormantCalculation(unittest.TestCase):
    """Formant shift parameter calculation logic tests (pure calculation extracted from _infer_impl)."""

    def test_formant_factor_zero(self):
        """formant=0 gives factor=1.0."""
        formant = 0.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 1.0)

    def test_formant_factor_positive(self):
        """formant=12 gives factor=2.0 (one octave up)."""
        formant = 12.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 2.0)

    def test_formant_factor_negative(self):
        """formant=-12 gives factor=0.5 (one octave down)."""
        formant = -12.0
        factor = pow(2, formant / 12)
        self.assertAlmostEqual(factor, 0.5)

    def test_return_length2_proportional_to_factor(self):
        """return_length2 is proportional to factor."""
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
