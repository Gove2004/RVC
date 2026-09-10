"""配置体系单元测试（unittest 风格，无需额外依赖）。"""
import unittest

from rvc.core.config import (
    AppConfig,
    EngineConfig,
    InferenceConfig,
    ModelEntry,
)


class TestInferenceConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = InferenceConfig()
        self.assertEqual(cfg.pitch, 0)
        self.assertEqual(cfg.formant, 0.0)
        self.assertEqual(cfg.protect, 0.5)
        self.assertEqual(cfg.f0_method, "rmvpe")
        self.assertEqual(cfg.rms_mix, 0.0)

    def test_mutation(self):
        cfg = InferenceConfig()
        cfg.pitch = 12
        cfg.formant = 1.5
        self.assertEqual(cfg.pitch, 12)
        self.assertEqual(cfg.formant, 1.5)


class TestEngineConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = EngineConfig()
        self.assertEqual(cfg.block_time, 0.25)
        self.assertEqual(cfg.crossfade_time, 0.05)
        self.assertEqual(cfg.extra_time, 2.5)
        self.assertEqual(cfg.sr_mode, "model")
        self.assertFalse(cfg.enable_out2)


class TestAppConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertIsInstance(cfg.inference, InferenceConfig)
        self.assertIsInstance(cfg.engine, EngineConfig)
        self.assertEqual(cfg.active_model, "")
        self.assertEqual(cfg.models, [])

    def test_nested_mutation(self):
        cfg = AppConfig()
        cfg.inference.pitch = 6
        cfg.engine.block_time = 0.5
        cfg.active_model = "/path/to/model.pth"
        self.assertEqual(cfg.inference.pitch, 6)
        self.assertEqual(cfg.engine.block_time, 0.5)
        self.assertEqual(cfg.active_model, "/path/to/model.pth")


class TestModelEntry(unittest.TestCase):
    def test_defaults(self):
        m = ModelEntry()
        self.assertEqual(m.name, "")
        self.assertEqual(m.pth, "")
        self.assertEqual(m.pitch, 0)
        self.assertEqual(m.formant, 0.0)
        self.assertEqual(m.hubert, "chinese")


if __name__ == "__main__":
    unittest.main()
