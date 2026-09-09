"""配置体系深度测试 — 所有 dataclass 的默认值、边界、序列化、继承。

覆盖：
- BreakProtectConfig / InferenceConfig / EngineConfig
- ModelEntry / AppConfig / OfflineConfig / TrainConfig
- 默认值验证、字段覆盖、嵌套配置、继承关系
"""
import copy
import unittest
from dataclasses import asdict, fields, is_dataclass

from rvc.core.config import (
    AppConfig,
    BreakProtectConfig,
    EngineConfig,
    HUBERT_DEFAULT,
    InferenceConfig,
    ModelEntry,
    OfflineConfig,
    TrainConfig,
)


class TestBreakProtectConfig(unittest.TestCase):
    """BreakProtectConfig 断点保护配置测试。"""

    def test_default_values(self):
        cfg = BreakProtectConfig()
        self.assertTrue(cfg.enable)
        self.assertAlmostEqual(cfg.src_hz, 300.0)
        self.assertAlmostEqual(cfg.ratio, 0.4)
        self.assertAlmostEqual(cfg.knee, 0.12)

    def test_custom_values(self):
        cfg = BreakProtectConfig(enable=False, src_hz=500.0, ratio=0.6, knee=0.2)
        self.assertFalse(cfg.enable)
        self.assertAlmostEqual(cfg.src_hz, 500.0)
        self.assertAlmostEqual(cfg.ratio, 0.6)
        self.assertAlmostEqual(cfg.knee, 0.2)

    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(BreakProtectConfig()))

    def test_field_count(self):
        self.assertEqual(len(fields(BreakProtectConfig)), 4)

    def test_asdict(self):
        cfg = BreakProtectConfig()
        d = asdict(cfg)
        self.assertEqual(set(d.keys()), {"enable", "src_hz", "ratio", "knee"})

    def test_equality(self):
        self.assertEqual(BreakProtectConfig(), BreakProtectConfig())
        self.assertNotEqual(BreakProtectConfig(), BreakProtectConfig(enable=False))

    def test_copy(self):
        cfg = BreakProtectConfig(src_hz=400.0)
        cfg2 = copy.copy(cfg)
        self.assertEqual(cfg, cfg2)
        cfg2.src_hz = 500.0
        self.assertNotEqual(cfg.src_hz, cfg2.src_hz)

    def test_deepcopy_independent(self):
        cfg = BreakProtectConfig()
        cfg2 = copy.deepcopy(cfg)
        cfg2.enable = False
        self.assertTrue(cfg.enable)

    def test_zero_values(self):
        cfg = BreakProtectConfig(enable=False, src_hz=0.0, ratio=0.0, knee=0.0)
        self.assertFalse(cfg.enable)
        self.assertAlmostEqual(cfg.src_hz, 0.0)

    def test_negative_values_accepted(self):
        """dataclass 不做校验，负值也能设置（由使用方校验）。"""
        cfg = BreakProtectConfig(src_hz=-100.0, ratio=-0.5)
        self.assertAlmostEqual(cfg.src_hz, -100.0)

    def test_large_values(self):
        cfg = BreakProtectConfig(src_hz=10000.0, ratio=100.0, knee=10.0)
        self.assertAlmostEqual(cfg.src_hz, 10000.0)


class TestInferenceConfig(unittest.TestCase):
    """InferenceConfig 推理配置测试。"""

    def test_default_values(self):
        cfg = InferenceConfig()
        self.assertEqual(cfg.pitch, 0)
        self.assertAlmostEqual(cfg.formant, 0.0)
        self.assertAlmostEqual(cfg.protect, 0.5)
        self.assertEqual(cfg.f0_method, "rmvpe")
        self.assertAlmostEqual(cfg.rms_mix, 0.0)

    def test_nested_break_protect_default(self):
        cfg = InferenceConfig()
        self.assertIsInstance(cfg.break_protect, BreakProtectConfig)
        self.assertTrue(cfg.break_protect.enable)

    def test_custom_nested_config(self):
        bp = BreakProtectConfig(enable=False, src_hz=400.0)
        cfg = InferenceConfig(break_protect=bp)
        self.assertFalse(cfg.break_protect.enable)
        self.assertAlmostEqual(cfg.break_protect.src_hz, 400.0)

    def test_field_count(self):
        self.assertEqual(len(fields(InferenceConfig)), 6)

    def test_asdict_nested(self):
        d = asdict(InferenceConfig())
        self.assertIn("break_protect", d)
        self.assertIsInstance(d["break_protect"], dict)

    def test_pitch_negative(self):
        cfg = InferenceConfig(pitch=-12)
        self.assertEqual(cfg.pitch, -12)

    def test_pitch_positive(self):
        cfg = InferenceConfig(pitch=12)
        self.assertEqual(cfg.pitch, 12)

    def test_formant_negative(self):
        cfg = InferenceConfig(formant=-0.5)
        self.assertAlmostEqual(cfg.formant, -0.5)

    def test_formant_positive(self):
        cfg = InferenceConfig(formant=0.5)
        self.assertAlmostEqual(cfg.formant, 0.5)

    def test_f0_method_variants(self):
        for method in ["rmvpe", "fcpe", "crepe", "pm", "harvest"]:
            cfg = InferenceConfig(f0_method=method)
            self.assertEqual(cfg.f0_method, method)

    def test_protect_zero(self):
        cfg = InferenceConfig(protect=0.0)
        self.assertAlmostEqual(cfg.protect, 0.0)

    def test_protect_one(self):
        cfg = InferenceConfig(protect=1.0)
        self.assertAlmostEqual(cfg.protect, 1.0)

    def test_rms_mix_zero(self):
        cfg = InferenceConfig(rms_mix=0.0)
        self.assertAlmostEqual(cfg.rms_mix, 0.0)

    def test_rms_mix_one(self):
        cfg = InferenceConfig(rms_mix=1.0)
        self.assertAlmostEqual(cfg.rms_mix, 1.0)

    def test_equality(self):
        self.assertEqual(InferenceConfig(), InferenceConfig())
        self.assertNotEqual(InferenceConfig(), InferenceConfig(pitch=1))

    def test_nested_equality(self):
        cfg1 = InferenceConfig()
        cfg2 = InferenceConfig()
        cfg2.break_protect.enable = False
        self.assertNotEqual(cfg1, cfg2)

    def test_deepcopy_nested_independent(self):
        cfg = InferenceConfig()
        cfg2 = copy.deepcopy(cfg)
        cfg2.break_protect.enable = False
        self.assertTrue(cfg.break_protect.enable)


class TestEngineConfig(unittest.TestCase):
    """EngineConfig 引擎配置测试。"""

    def test_default_values(self):
        cfg = EngineConfig()
        self.assertAlmostEqual(cfg.block_time, 0.25)
        self.assertAlmostEqual(cfg.crossfade_time, 0.05)
        self.assertAlmostEqual(cfg.extra_time, 2.5)
        self.assertEqual(cfg.sr_mode, "model")
        self.assertEqual(cfg.hostapi, "")
        self.assertEqual(cfg.input_device, "")
        self.assertEqual(cfg.output_device, "")
        self.assertEqual(cfg.output2_device, "")
        self.assertFalse(cfg.enable_out2)

    def test_custom_values(self):
        cfg = EngineConfig(
            block_time=0.1, crossfade_time=0.02, extra_time=1.0,
            sr_mode="fixed", hostapi="MME", input_device="mic",
            output_device="speaker", output2_device="headphones",
            enable_out2=True,
        )
        self.assertAlmostEqual(cfg.block_time, 0.1)
        self.assertTrue(cfg.enable_out2)
        self.assertEqual(cfg.input_device, "mic")

    def test_field_count(self):
        self.assertEqual(len(fields(EngineConfig)), 9)

    def test_asdict(self):
        d = asdict(EngineConfig())
        self.assertEqual(len(d), 9)

    def test_equality(self):
        self.assertEqual(EngineConfig(), EngineConfig())
        self.assertNotEqual(EngineConfig(), EngineConfig(enable_out2=True))

    def test_block_time_zero(self):
        cfg = EngineConfig(block_time=0.0)
        self.assertAlmostEqual(cfg.block_time, 0.0)

    def test_sr_mode_variants(self):
        for mode in ["model", "fixed", "auto"]:
            cfg = EngineConfig(sr_mode=mode)
            self.assertEqual(cfg.sr_mode, mode)

    def test_enable_out2_without_device(self):
        cfg = EngineConfig(enable_out2=True, output2_device="")
        self.assertTrue(cfg.enable_out2)
        self.assertEqual(cfg.output2_device, "")


class TestModelEntry(unittest.TestCase):
    """ModelEntry 模型条目测试。"""

    def test_default_values(self):
        m = ModelEntry()
        self.assertEqual(m.name, "")
        self.assertEqual(m.pth, "")
        self.assertEqual(m.pitch, 0)
        self.assertAlmostEqual(m.formant, 0.0)
        self.assertEqual(m.hubert, "chinese")

    def test_custom_values(self):
        m = ModelEntry(name="test", pth="/path/model.pth", pitch=5, formant=0.3, hubert="base")
        self.assertEqual(m.name, "test")
        self.assertEqual(m.pitch, 5)

    def test_field_count(self):
        self.assertEqual(len(fields(ModelEntry)), 5)

    def test_asdict(self):
        d = asdict(ModelEntry())
        self.assertEqual(set(d.keys()), {"name", "pth", "pitch", "formant", "hubert"})

    def test_equality(self):
        self.assertEqual(ModelEntry(), ModelEntry())
        self.assertNotEqual(ModelEntry(), ModelEntry(name="x"))

    def test_hubert_default_constant(self):
        m = ModelEntry()
        self.assertEqual(m.hubert, HUBERT_DEFAULT)


class TestAppConfig(unittest.TestCase):
    """AppConfig 应用配置测试。"""

    def test_default_values(self):
        cfg = AppConfig()
        self.assertIsInstance(cfg.inference, InferenceConfig)
        self.assertIsInstance(cfg.engine, EngineConfig)
        self.assertEqual(cfg.active_model, "")
        self.assertEqual(cfg.models, [])

    def test_custom_models(self):
        models = [ModelEntry(name="a"), ModelEntry(name="b")]
        cfg = AppConfig(models=models)
        self.assertEqual(len(cfg.models), 2)
        self.assertEqual(cfg.models[0].name, "a")

    def test_field_count(self):
        self.assertEqual(len(fields(AppConfig)), 4)

    def test_asdict_nested(self):
        d = asdict(AppConfig())
        self.assertIn("inference", d)
        self.assertIn("engine", d)
        self.assertIn("models", d)
        self.assertIsInstance(d["inference"], dict)
        self.assertIsInstance(d["models"], list)

    def test_equality(self):
        self.assertEqual(AppConfig(), AppConfig())
        self.assertNotEqual(AppConfig(), AppConfig(active_model="x"))

    def test_models_list_independent(self):
        cfg1 = AppConfig()
        cfg2 = copy.deepcopy(cfg1)
        cfg2.models.append(ModelEntry(name="new"))
        self.assertEqual(len(cfg1.models), 0)
        self.assertEqual(len(cfg2.models), 1)

    def test_active_model_set(self):
        cfg = AppConfig(active_model="model_a")
        self.assertEqual(cfg.active_model, "model_a")


class TestOfflineConfig(unittest.TestCase):
    """OfflineConfig 离线推理配置测试（继承 InferenceConfig）。"""

    def test_inherits_inference_fields(self):
        cfg = OfflineConfig()
        self.assertEqual(cfg.pitch, 0)
        self.assertEqual(cfg.f0_method, "rmvpe")
        self.assertIsInstance(cfg.break_protect, BreakProtectConfig)

    def test_offline_specific_defaults(self):
        cfg = OfflineConfig()
        self.assertEqual(cfg.input_path, "")
        self.assertEqual(cfg.output_path, "")
        self.assertEqual(cfg.model_path, "")
        self.assertEqual(cfg.hubert, HUBERT_DEFAULT)

    def test_custom_values(self):
        cfg = OfflineConfig(
            input_path="/in.wav", output_path="/out.wav",
            model_path="/model.pth", hubert="base", pitch=3,
        )
        self.assertEqual(cfg.input_path, "/in.wav")
        self.assertEqual(cfg.pitch, 3)
        self.assertEqual(cfg.hubert, "base")

    def test_is_inference_config_subclass(self):
        self.assertTrue(issubclass(OfflineConfig, InferenceConfig))

    def test_field_count_includes_inherited(self):
        """OfflineConfig 字段 = InferenceConfig 6 个 + 4 个特有 = 10 个。"""
        self.assertEqual(len(fields(OfflineConfig)), 10)

    def test_asdict_includes_all(self):
        d = asdict(OfflineConfig())
        self.assertIn("pitch", d)  # 继承的
        self.assertIn("input_path", d)  # 特有的
        self.assertIn("break_protect", d)  # 嵌套的

    def test_equality(self):
        self.assertEqual(OfflineConfig(), OfflineConfig())
        self.assertNotEqual(OfflineConfig(), OfflineConfig(input_path="x"))

    def test_hubert_override(self):
        cfg = OfflineConfig(hubert="japanese")
        self.assertEqual(cfg.hubert, "japanese")


class TestTrainConfig(unittest.TestCase):
    """TrainConfig 训练配置测试。"""

    def test_required_exp_dir(self):
        """exp_dir 是必填字段，无默认值。"""
        with self.assertRaises(TypeError):
            TrainConfig()

    def test_default_values(self):
        cfg = TrainConfig(exp_dir="/tmp/exp")
        self.assertEqual(cfg.exp_dir, "/tmp/exp")
        self.assertEqual(cfg.sr, 48000)
        self.assertEqual(cfg.epochs, 2000)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.save_every_epoch, 200)
        self.assertAlmostEqual(cfg.learning_rate, 1e-4)
        self.assertEqual(cfg.pretrain_g, "")
        self.assertEqual(cfg.pretrain_d, "")
        self.assertTrue(cfg.fp16_run)
        self.assertEqual(cfg.device, "cuda:0")
        self.assertEqual(cfg.log_interval, 20)
        self.assertEqual(cfg.keep_ckpts, 1)
        self.assertEqual(cfg.keep_models, 0)

    def test_custom_values(self):
        cfg = TrainConfig(
            exp_dir="/exp", sr=16000, epochs=100, batch_size=8,
            save_every_epoch=50, learning_rate=1e-3, fp16_run=False,
            device="cpu", log_interval=10, keep_ckpts=5, keep_models=3,
        )
        self.assertEqual(cfg.sr, 16000)
        self.assertEqual(cfg.epochs, 100)
        self.assertFalse(cfg.fp16_run)
        self.assertEqual(cfg.device, "cpu")

    def test_field_count(self):
        self.assertEqual(len(fields(TrainConfig)), 13)

    def test_asdict(self):
        d = asdict(TrainConfig(exp_dir="/exp"))
        self.assertEqual(d["exp_dir"], "/exp")
        self.assertEqual(len(d), 13)

    def test_equality(self):
        self.assertEqual(TrainConfig(exp_dir="/a"), TrainConfig(exp_dir="/a"))
        self.assertNotEqual(TrainConfig(exp_dir="/a"), TrainConfig(exp_dir="/b"))

    def test_sr_variants(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            cfg = TrainConfig(exp_dir="/exp", sr=sr)
            self.assertEqual(cfg.sr, sr)

    def test_device_variants(self):
        for dev in ["cpu", "cuda:0", "cuda:1", "mps"]:
            cfg = TrainConfig(exp_dir="/exp", device=dev)
            self.assertEqual(cfg.device, dev)

    def test_learning_rate_small(self):
        cfg = TrainConfig(exp_dir="/exp", learning_rate=1e-6)
        self.assertAlmostEqual(cfg.learning_rate, 1e-6)

    def test_learning_rate_large(self):
        cfg = TrainConfig(exp_dir="/exp", learning_rate=1.0)
        self.assertAlmostEqual(cfg.learning_rate, 1.0)

    def test_batch_size_one(self):
        cfg = TrainConfig(exp_dir="/exp", batch_size=1)
        self.assertEqual(cfg.batch_size, 1)

    def test_epochs_zero(self):
        cfg = TrainConfig(exp_dir="/exp", epochs=0)
        self.assertEqual(cfg.epochs, 0)

    def test_keep_models_zero(self):
        cfg = TrainConfig(exp_dir="/exp", keep_models=0)
        self.assertEqual(cfg.keep_models, 0)

    def test_pretrain_paths(self):
        cfg = TrainConfig(exp_dir="/exp", pretrain_g="/g.pth", pretrain_d="/d.pth")
        self.assertEqual(cfg.pretrain_g, "/g.pth")
        self.assertEqual(cfg.pretrain_d, "/d.pth")


class TestHubertDefaultConstant(unittest.TestCase):
    """HUBERT_DEFAULT 常量测试。"""

    def test_value(self):
        self.assertEqual(HUBERT_DEFAULT, "chinese")

    def test_used_in_model_entry(self):
        self.assertEqual(ModelEntry().hubert, HUBERT_DEFAULT)

    def test_used_in_offline_config(self):
        self.assertEqual(OfflineConfig().hubert, HUBERT_DEFAULT)


if __name__ == "__main__":
    unittest.main()
