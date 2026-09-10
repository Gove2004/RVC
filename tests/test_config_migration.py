"""配置持久化迁移单元测试 — 旧短键格式 → 新嵌套结构。

验证 _migrate_old_format 的检测逻辑、state_from_dict/state_to_dict 的读写、
以及新旧格式的往返一致性。
"""
import unittest

from gui.infer.viewmodel.param_binding import (
    _migrate_old_format,
    state_from_dict,
    state_to_dict,
)

# 典型的旧短键格式（来自实际 save_state.json）
OLD_FORMAT = {
    "protect": 0.8,
    "f0": "rmvpe",
    "rms": 0.5,
    "bl": 0.1,
    "cf": 0.15,
    "ex": 5.0,
    "sr_mode": "device",
    "ha": "Windows WASAPI",
    "in_dev": "麦克风 (USB Audio Device)",
    "out_dev": "CABLE Input (VB-Audio Virtual Cable)",
    "out2_dev": "扬声器 (USB Audio Device)",
    "active_model": "E:/Projects/Python/RVC/assets/models/test.pth",
}

# 对应的新嵌套结构
NEW_FORMAT = {
    "inference": {
        "protect": 0.8,
        "f0_method": "rmvpe",
        "rms_mix": 0.5,
    },
    "engine": {
        "block_time": 0.1,
        "crossfade_time": 0.15,
        "extra_time": 5.0,
        "sr_mode": "device",
        "hostapi": "Windows WASAPI",
        "input_device": "麦克风 (USB Audio Device)",
        "output_device": "CABLE Input (VB-Audio Virtual Cable)",
        "output2_device": "扬声器 (USB Audio Device)",
    },
    "active_model": "E:/Projects/Python/RVC/assets/models/test.pth",
}


class TestMigrateOldFormat(unittest.TestCase):
    def test_old_format_migrates_to_nested(self):
        result = _migrate_old_format(OLD_FORMAT)
        # 验证嵌套结构
        self.assertEqual(result["inference"]["protect"], 0.8)
        self.assertEqual(result["inference"]["f0_method"], "rmvpe")
        self.assertEqual(result["inference"]["rms_mix"], 0.5)
        self.assertEqual(result["engine"]["block_time"], 0.1)
        self.assertEqual(result["engine"]["crossfade_time"], 0.15)
        self.assertEqual(result["engine"]["extra_time"], 5.0)
        self.assertEqual(result["engine"]["sr_mode"], "device")
        self.assertEqual(result["engine"]["hostapi"], "Windows WASAPI")
        self.assertEqual(result["engine"]["input_device"], "麦克风 (USB Audio Device)")
        self.assertEqual(result["engine"]["output_device"], "CABLE Input (VB-Audio Virtual Cable)")
        self.assertEqual(result["engine"]["output2_device"], "扬声器 (USB Audio Device)")
        self.assertEqual(result["active_model"], "E:/Projects/Python/RVC/assets/models/test.pth")
        # 旧短键应该被移除
        self.assertNotIn("bl", result)

    def test_new_format_not_migrated(self):
        # 新格式只有 active_model 一个键可能与旧键重合，不应触发迁移
        result = _migrate_old_format(NEW_FORMAT)
        self.assertIs(result, NEW_FORMAT)  # 原样返回（同一个对象）

    def test_partial_old_keys_no_migration(self):
        # 只有 1-2 个旧键（比如只有 active_model），不应触发迁移
        data = {"active_model": "test.pth", "inference": {"protect": 0.5}}
        result = _migrate_old_format(data)
        self.assertIs(result, data)

    def test_migration_preserves_extra_keys(self):
        # 迁移时应保留非配置字段（比如未来新增的字段）
        data = dict(OLD_FORMAT)
        data["custom_field"] = "hello"
        result = _migrate_old_format(data)
        self.assertEqual(result["custom_field"], "hello")

    def test_empty_dict_no_migration(self):
        result = _migrate_old_format({})
        self.assertEqual(result, {})


class TestStateFromDict(unittest.TestCase):
    def test_old_format_reads_correctly(self):
        cfg = state_from_dict(OLD_FORMAT)
        self.assertEqual(cfg.inference.protect, 0.8)
        self.assertEqual(cfg.inference.f0_method, "rmvpe")
        self.assertEqual(cfg.inference.rms_mix, 0.5)
        self.assertEqual(cfg.engine.block_time, 0.1)
        self.assertEqual(cfg.engine.crossfade_time, 0.15)
        self.assertEqual(cfg.engine.extra_time, 5.0)
        self.assertEqual(cfg.engine.sr_mode, "device")
        self.assertEqual(cfg.engine.hostapi, "Windows WASAPI")
        self.assertEqual(cfg.engine.input_device, "麦克风 (USB Audio Device)")
        self.assertEqual(cfg.engine.output_device, "CABLE Input (VB-Audio Virtual Cable)")
        self.assertEqual(cfg.engine.output2_device, "扬声器 (USB Audio Device)")
        self.assertEqual(cfg.active_model, "E:/Projects/Python/RVC/assets/models/test.pth")

    def test_new_format_reads_correctly(self):
        cfg = state_from_dict(NEW_FORMAT)
        self.assertEqual(cfg.inference.protect, 0.8)
        self.assertEqual(cfg.inference.f0_method, "rmvpe")
        self.assertEqual(cfg.engine.block_time, 0.1)
        self.assertEqual(cfg.active_model, "E:/Projects/Python/RVC/assets/models/test.pth")

    def test_empty_dict_uses_defaults(self):
        cfg = state_from_dict({})
        # 验证用的是 AppConfig 默认值
        self.assertEqual(cfg.inference.protect, 0.5)
        self.assertEqual(cfg.inference.f0_method, "rmvpe")
        self.assertEqual(cfg.inference.rms_mix, 0.0)
        self.assertEqual(cfg.engine.block_time, 0.25)
        self.assertEqual(cfg.engine.crossfade_time, 0.05)
        self.assertEqual(cfg.engine.extra_time, 2.5)

    def test_enable_out2_derived_from_output2_device(self):
        # 有副输出设备时 enable_out2 = True
        cfg = state_from_dict(OLD_FORMAT)
        self.assertTrue(cfg.engine.enable_out2)
        # 没有副输出设备时 enable_out2 = False
        data = dict(NEW_FORMAT)
        data["engine"]["output2_device"] = "不使用"
        cfg = state_from_dict(data)
        self.assertFalse(cfg.engine.enable_out2)


class TestStateToDict(unittest.TestCase):
    def test_outputs_nested_structure(self):
        cfg = state_from_dict(OLD_FORMAT)
        result = state_to_dict(cfg)
        # 验证是嵌套结构，不是短键
        self.assertIn("inference", result)
        self.assertIn("engine", result)
        self.assertIn("active_model", result)
        # 不应有旧短键
        self.assertNotIn("bl", result)

    def test_values_preserved(self):
        cfg = state_from_dict(OLD_FORMAT)
        result = state_to_dict(cfg)
        self.assertEqual(result["inference"]["protect"], 0.8)
        self.assertEqual(result["inference"]["f0_method"], "rmvpe")
        self.assertEqual(result["engine"]["block_time"], 0.1)
        self.assertEqual(result["active_model"], "E:/Projects/Python/RVC/assets/models/test.pth")


class TestRoundTripConsistency(unittest.TestCase):
    def test_old_format_roundtrip(self):
        """旧格式 → AppConfig → 新格式 → AppConfig，关键值一致。"""
        cfg1 = state_from_dict(OLD_FORMAT)
        new_dict = state_to_dict(cfg1)
        cfg2 = state_from_dict(new_dict)
        # 验证关键值一致
        self.assertEqual(cfg1.inference.protect, cfg2.inference.protect)
        self.assertEqual(cfg1.inference.f0_method, cfg2.inference.f0_method)
        self.assertEqual(cfg1.inference.rms_mix, cfg2.inference.rms_mix)
        self.assertEqual(cfg1.engine.block_time, cfg2.engine.block_time)
        self.assertEqual(cfg1.engine.crossfade_time, cfg2.engine.crossfade_time)
        self.assertEqual(cfg1.engine.extra_time, cfg2.engine.extra_time)
        self.assertEqual(cfg1.engine.sr_mode, cfg2.engine.sr_mode)
        self.assertEqual(cfg1.engine.hostapi, cfg2.engine.hostapi)
        self.assertEqual(cfg1.engine.input_device, cfg2.engine.input_device)
        self.assertEqual(cfg1.engine.output_device, cfg2.engine.output_device)
        self.assertEqual(cfg1.engine.output2_device, cfg2.engine.output2_device)
        self.assertEqual(cfg1.active_model, cfg2.active_model)

    def test_new_format_roundtrip(self):
        """新格式 → AppConfig → 新格式 → AppConfig，关键值一致。"""
        cfg1 = state_from_dict(NEW_FORMAT)
        new_dict = state_to_dict(cfg1)
        cfg2 = state_from_dict(new_dict)
        self.assertEqual(cfg1.inference.protect, cfg2.inference.protect)
        self.assertEqual(cfg1.engine.block_time, cfg2.engine.block_time)
        self.assertEqual(cfg1.active_model, cfg2.active_model)


if __name__ == "__main__":
    unittest.main()
