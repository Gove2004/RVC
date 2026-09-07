"""配置体系单元测试（unittest 风格，无需额外依赖）。"""
import json
import tempfile
import unittest
from pathlib import Path

from rvc.config import (
    AppConfig,
    BreakProtectConfig,
    DenoiseConfig,
    EngineConfig,
    InferenceConfig,
    ModelEntry,
    OfflineTask,
    config_to_dict,
    dict_to_config,
    load_config_json,
    save_config_json,
)


class TestInferenceConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = InferenceConfig()
        self.assertEqual(cfg.pitch, 0)
        self.assertEqual(cfg.formant, 0.0)
        self.assertEqual(cfg.protect, 0.5)
        self.assertEqual(cfg.f0_method, "rmvpe")
        self.assertEqual(cfg.rms_mix, 0.0)
        self.assertEqual(cfg.hubert_variant, "chinese")

    def test_nested_defaults(self):
        cfg = InferenceConfig()
        self.assertIsInstance(cfg.break_protect, BreakProtectConfig)
        self.assertTrue(cfg.break_protect.enable)
        self.assertEqual(cfg.break_protect.src_hz, 300.0)
        self.assertEqual(cfg.break_protect.ratio, 0.4)
        self.assertEqual(cfg.break_protect.knee, 0.12)
        self.assertIsInstance(cfg.denoise, DenoiseConfig)
        self.assertFalse(cfg.denoise.enable)
        self.assertEqual(cfg.denoise.strength, 0.5)

    def test_mutation(self):
        cfg = InferenceConfig()
        cfg.pitch = 12
        cfg.formant = 1.5
        cfg.break_protect.enable = False
        cfg.denoise.enable = True
        self.assertEqual(cfg.pitch, 12)
        self.assertEqual(cfg.formant, 1.5)
        self.assertFalse(cfg.break_protect.enable)
        self.assertTrue(cfg.denoise.enable)


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


class TestSerialization(unittest.TestCase):
    def test_config_to_dict(self):
        cfg = InferenceConfig(pitch=12, formant=1.0)
        d = config_to_dict(cfg)
        self.assertEqual(d["pitch"], 12)
        self.assertEqual(d["formant"], 1.0)
        self.assertTrue(d["break_protect"]["enable"])
        self.assertFalse(d["denoise"]["enable"])

    def test_dict_to_config(self):
        d = {"pitch": 6, "formant": 0.5, "f0_method": "fcpe"}
        cfg = dict_to_config(d, InferenceConfig)
        self.assertEqual(cfg.pitch, 6)
        self.assertEqual(cfg.formant, 0.5)
        self.assertEqual(cfg.f0_method, "fcpe")
        self.assertEqual(cfg.protect, 0.5)  # 缺省字段用默认值

    def test_json_roundtrip(self):
        cfg = AppConfig()
        cfg.inference.pitch = 12
        cfg.engine.block_time = 0.5
        cfg.active_model = "/test/model.pth"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config_json(cfg, path)
            loaded = load_config_json(path, AppConfig)
            self.assertEqual(loaded.inference.pitch, 12)
            self.assertEqual(loaded.engine.block_time, 0.5)
            self.assertEqual(loaded.active_model, "/test/model.pth")

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            with self.assertRaises(FileNotFoundError):
                load_config_json(path, AppConfig)

    def test_load_corrupted_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupted.json"
            path.write_text("{invalid json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_config_json(path, AppConfig)


class TestModelEntry(unittest.TestCase):
    def test_defaults(self):
        m = ModelEntry()
        self.assertEqual(m.name, "")
        self.assertEqual(m.pth, "")
        self.assertEqual(m.pitch, 0)
        self.assertEqual(m.formant, 0.0)
        self.assertEqual(m.hubert, "chinese")


class TestOfflineTask(unittest.TestCase):
    def test_defaults(self):
        t = OfflineTask()
        self.assertEqual(t.input_path, "")
        self.assertEqual(t.output_path, "")
        self.assertEqual(t.model_path, "")
        self.assertEqual(t.pad_sec, 3.0)
        self.assertIsInstance(t.inference, InferenceConfig)
        self.assertIsInstance(t.engine, EngineConfig)


if __name__ == "__main__":
    unittest.main()
