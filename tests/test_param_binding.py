"""param_binding 纯函数测试 — 嵌套访问、状态序列化、gender/formant 转换、错误格式化。"""
import unittest

from rvc.config import AppConfig, InferenceConfig, EngineConfig
from gui.infer.param_binding import (
    _get_nested, _set_nested, _has_nested,
    _get_nested_dict, _set_nested_dict,
    _migrate_old_format, _parse,
    state_from_dict, state_to_dict,
    runtime_from_state, engine_from_state,
    gender_to_formant, formant_to_gender,
    format_error_message,
    CHECK, X100, INT, COMBO, TEXT,
)


class TestNestedAccess(unittest.TestCase):
    """嵌套属性访问测试。"""

    def test_get_nested_simple(self):
        """简单属性访问。"""
        from types import SimpleNamespace
        obj = SimpleNamespace(a=1, b=SimpleNamespace(c=2))
        self.assertEqual(_get_nested(obj, "a"), 1)
        self.assertEqual(_get_nested(obj, "b.c"), 2)

    def test_set_nested(self):
        """设置嵌套属性。"""
        from types import SimpleNamespace
        obj = SimpleNamespace(a=SimpleNamespace(b=1))
        _set_nested(obj, "a.b", 42)
        self.assertEqual(obj.a.b, 42)

    def test_has_nested_dict(self):
        """_has_nested 检查 dict 嵌套路径。"""
        d = {"a": {"b": {"c": 1}}}
        self.assertTrue(_has_nested(d, "a"))
        self.assertTrue(_has_nested(d, "a.b"))
        self.assertTrue(_has_nested(d, "a.b.c"))
        self.assertFalse(_has_nested(d, "x"))
        self.assertFalse(_has_nested(d, "a.x"))
        self.assertFalse(_has_nested(d, "a.b.x"))

    def test_has_nested_non_dict_returns_false(self):
        """非 dict 对象应返回 False。"""
        self.assertFalse(_has_nested("not a dict", "a"))
        self.assertFalse(_has_nested(42, "a"))

    def test_get_nested_dict(self):
        """嵌套字典访问。"""
        d = {"a": {"b": {"c": 42}}}
        self.assertEqual(_get_nested_dict(d, "a.b.c"), 42)
        self.assertEqual(_get_nested_dict(d, "a.b"), {"c": 42})

    def test_set_nested_dict(self):
        """设置嵌套字典。"""
        d = {"a": {}}
        _set_nested_dict(d, "a.b.c", 42)
        self.assertEqual(d["a"]["b"]["c"], 42)

    def test_set_nested_dict_creates_intermediate(self):
        """设置时应自动创建中间字典。"""
        d = {}
        _set_nested_dict(d, "a.b.c", 42)
        self.assertEqual(d, {"a": {"b": {"c": 42}}})


class TestParse(unittest.TestCase):
    """_parse 类型解析测试。"""

    def test_parse_check(self):
        """CHECK 类型用 bool() 转换。"""
        self.assertTrue(_parse(CHECK, True))
        self.assertFalse(_parse(CHECK, False))
        self.assertTrue(_parse(CHECK, "True"))  # 非空字符串 → True
        self.assertTrue(_parse(CHECK, 1))
        self.assertFalse(_parse(CHECK, 0))

    def test_parse_x100(self):
        """X100 类型转 float。"""
        self.assertAlmostEqual(_parse(X100, "0.5"), 0.5)
        self.assertAlmostEqual(_parse(X100, 50), 50.0)

    def test_parse_int(self):
        """INT 类型转 int。"""
        self.assertEqual(_parse(INT, "42"), 42)
        self.assertEqual(_parse(INT, -5), -5)

    def test_parse_combo_text(self):
        """COMBO/TEXT 类型转 str。"""
        self.assertEqual(_parse(COMBO, "rmvpe"), "rmvpe")
        self.assertEqual(_parse(TEXT, "hello"), "hello")


class TestStateSerialization(unittest.TestCase):
    """state_from_dict / state_to_dict 往返测试。

    注意：pitch/formant 是模型卡片级参数，不在全局 BINDINGS 表中，
    由 model_manager 管理持久化。这里只测试 BINDINGS 表中的字段。
    """

    def test_roundtrip_default(self):
        """默认配置应能往返序列化。"""
        state = AppConfig()
        d = state_to_dict(state)
        state2 = state_from_dict(d)
        self.assertEqual(state.inference.protect, state2.inference.protect)
        self.assertEqual(state.inference.f0_method, state2.inference.f0_method)
        self.assertEqual(state.inference.rms_mix, state2.inference.rms_mix)
        self.assertEqual(state.inference.denoise.enable, state2.inference.denoise.enable)
        self.assertEqual(state.inference.denoise.strength, state2.inference.denoise.strength)
        self.assertEqual(state.inference.break_protect.enable, state2.inference.break_protect.enable)
        self.assertEqual(state.inference.break_protect.src_hz, state2.inference.break_protect.src_hz)
        self.assertEqual(state.engine.block_time, state2.engine.block_time)
        self.assertEqual(state.engine.crossfade_time, state2.engine.crossfade_time)
        self.assertEqual(state.engine.extra_time, state2.engine.extra_time)
        self.assertEqual(state.active_model, state2.active_model)

    def test_roundtrip_custom(self):
        """自定义配置应能往返序列化。"""
        state = AppConfig()
        state.inference.protect = 0.3
        state.inference.f0_method = "fcpe"
        state.inference.rms_mix = 0.2
        state.inference.denoise.enable = True
        state.inference.denoise.strength = 0.8
        state.inference.break_protect.enable = False
        state.inference.break_protect.src_hz = 500.0
        state.engine.block_time = 0.5
        state.engine.crossfade_time = 0.1
        state.engine.extra_time = 1.0
        state.active_model = "test_model"
        d = state_to_dict(state)
        state2 = state_from_dict(d)
        self.assertEqual(state2.inference.protect, 0.3)
        self.assertEqual(state2.inference.f0_method, "fcpe")
        self.assertEqual(state2.inference.rms_mix, 0.2)
        self.assertTrue(state2.inference.denoise.enable)
        self.assertEqual(state2.inference.denoise.strength, 0.8)
        self.assertFalse(state2.inference.break_protect.enable)
        self.assertEqual(state2.inference.break_protect.src_hz, 500.0)
        self.assertEqual(state2.engine.block_time, 0.5)
        self.assertEqual(state2.engine.crossfade_time, 0.1)
        self.assertEqual(state2.engine.extra_time, 1.0)
        self.assertEqual(state2.active_model, "test_model")

    def test_dict_has_nested_structure(self):
        """序列化后的字典应有嵌套结构。"""
        state = AppConfig()
        d = state_to_dict(state)
        self.assertIn("inference", d)
        self.assertIn("engine", d)
        self.assertIn("protect", d["inference"])
        self.assertIn("f0_method", d["inference"])
        self.assertIn("denoise", d["inference"])
        self.assertIn("break_protect", d["inference"])
        self.assertIn("block_time", d["engine"])
        self.assertIn("active_model", d)

    def test_pitch_not_in_bindings(self):
        """pitch/formant 不在全局 BINDINGS 表中（模型卡片级参数）。"""
        state = AppConfig()
        state.inference.pitch = 5
        state.inference.formant = 1.5
        d = state_to_dict(state)
        # pitch/formant 不会被序列化到全局配置
        self.assertNotIn("pitch", d.get("inference", {}))
        self.assertNotIn("formant", d.get("inference", {}))


class TestMigrateOldFormat(unittest.TestCase):
    """_migrate_old_format 旧格式迁移测试。"""

    def test_new_format_unchanged(self):
        """新格式（嵌套结构）应不变。"""
        data = {"inference": {"protect": 0.5}, "engine": {"block_time": 0.25}}
        result = _migrate_old_format(data)
        self.assertEqual(result, data)

    def test_old_format_migrated(self):
        """旧格式（≥3 个短键）应迁移到嵌套结构。"""
        data = {"protect": 0.3, "f0": "fcpe", "rms": 0.2, "nr_en": True}
        result = _migrate_old_format(data)
        self.assertIn("inference", result)
        self.assertEqual(result["inference"]["protect"], 0.3)
        self.assertEqual(result["inference"]["f0_method"], "fcpe")
        self.assertEqual(result["inference"]["rms_mix"], 0.2)
        self.assertTrue(result["inference"]["denoise"]["enable"])

    def test_few_old_keys_unchanged(self):
        """少于 3 个旧短键应不变（可能是新格式的 active_model）。"""
        data = {"active_model": "test"}
        result = _migrate_old_format(data)
        self.assertEqual(result, data)

    def test_old_keys_removed_after_migration(self):
        """迁移后旧短键应被移除。"""
        data = {"protect": 0.3, "f0": "fcpe", "rms": 0.2}
        result = _migrate_old_format(data)
        self.assertNotIn("protect", result)
        self.assertNotIn("f0", result)
        self.assertNotIn("rms", result)


class TestStateConverters(unittest.TestCase):
    """runtime_from_state / engine_from_state 测试。"""

    def test_runtime_from_state(self):
        """应从 AppConfig 提取 InferenceConfig。"""
        state = AppConfig()
        state.inference.protect = 0.3
        state.inference.f0_method = "fcpe"
        runtime = runtime_from_state(state)
        self.assertIsInstance(runtime, InferenceConfig)
        self.assertEqual(runtime.protect, 0.3)
        self.assertEqual(runtime.f0_method, "fcpe")

    def test_engine_from_state(self):
        """应从 AppConfig 提取 EngineConfig。"""
        state = AppConfig()
        state.engine.block_time = 0.5
        state.engine.crossfade_time = 0.1
        engine = engine_from_state(state)
        self.assertIsInstance(engine, EngineConfig)
        self.assertEqual(engine.block_time, 0.5)
        self.assertEqual(engine.crossfade_time, 0.1)


class TestGenderFormant(unittest.TestCase):
    """gender_to_formant / formant_to_gender 往返测试。

    gender 滑杆范围 [0, 1]，formant 范围 [-2.5, +2.5]。
    gender=0.5 对应 formant=0。
    """

    def test_roundtrip(self):
        """gender → formant → gender 应往返一致。"""
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            f = gender_to_formant(v)
            v2 = formant_to_gender(f)
            self.assertAlmostEqual(v, v2, places=5)

    def test_midpoint_maps_to_zero(self):
        """gender=0.5 应映射到 formant=0。"""
        self.assertEqual(gender_to_formant(0.5), 0.0)
        self.assertEqual(formant_to_gender(0.0), 0.5)

    def test_positive_gender(self):
        """gender > 0.5 应映射到正 formant。"""
        self.assertGreater(gender_to_formant(0.75), 0)

    def test_negative_gender(self):
        """gender < 0.5 应映射到负 formant。"""
        self.assertLess(gender_to_formant(0.25), 0)

    def test_full_range(self):
        """gender=0 → formant=-2.5，gender=1 → formant=2.5。"""
        self.assertEqual(gender_to_formant(0.0), -2.5)
        self.assertEqual(gender_to_formant(1.0), 2.5)


class TestFormatErrorMessage(unittest.TestCase):
    """format_error_message 测试。"""

    def test_exception(self):
        """异常应返回其消息。"""
        e = ValueError("test error")
        self.assertEqual(format_error_message(e), "test error")

    def test_string(self):
        """字符串应原样返回。"""
        self.assertEqual(format_error_message("test error"), "test error")

    def test_empty_exception_message(self):
        """空消息异常应返回默认消息。"""
        e = ValueError()
        result = format_error_message(e)
        self.assertEqual(result, "未知错误")

    def test_multiline_error_keeps_last_line(self):
        """多行错误应只保留最后一行。"""
        e = Exception("line1\nline2\nline3")
        result = format_error_message(e)
        self.assertEqual(result, "line3")

    def test_none(self):
        """None 应返回空字符串或默认消息。"""
        result = format_error_message(None)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
