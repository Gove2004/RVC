"""GUI 配置持久化深度测试 — load_config/save_config/默认状态/坏 JSON 备份。

覆盖：
- _DEFAULT_STATE 结构验证
- load_config: 文件不存在返回默认、文件存在返回内容、坏 JSON 备份并返回默认
- save_config: 原子写、创建目录、内容正确
- 注意：使用临时文件和 mock 来测试，不影响真实配置
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gui.configs.config import (
    _DEFAULT_STATE,
    load_config,
    save_config,
)


class TestDefaultState(unittest.TestCase):
    """_DEFAULT_STATE 结构测试。"""

    def test_is_dict(self):
        self.assertIsInstance(_DEFAULT_STATE, dict)

    def test_has_gui_key(self):
        self.assertIn("gui", _DEFAULT_STATE)

    def test_has_train_key(self):
        self.assertIn("train", _DEFAULT_STATE)

    def test_has_models_key(self):
        self.assertIn("models", _DEFAULT_STATE)

    def test_gui_is_dict(self):
        self.assertIsInstance(_DEFAULT_STATE["gui"], dict)

    def test_train_is_dict(self):
        self.assertIsInstance(_DEFAULT_STATE["train"], dict)

    def test_models_is_list(self):
        self.assertIsInstance(_DEFAULT_STATE["models"], list)

    def test_models_empty(self):
        self.assertEqual(len(_DEFAULT_STATE["models"]), 0)

    def test_default_state_shallow_copy(self):
        """load_config 返回浅拷贝：顶层是新 dict，嵌套字典仍是引用（源代码行为）。"""
        state = dict(_DEFAULT_STATE)
        state["new_top_key"] = "value"
        self.assertNotIn("new_top_key", _DEFAULT_STATE)
        # 嵌套字典是共享引用（dict() 是浅拷贝）
        state["gui"]["shared"] = True
        self.assertIn("shared", _DEFAULT_STATE["gui"])
        # 清理
        del _DEFAULT_STATE["gui"]["shared"]


class TestLoadConfigMissingFile(unittest.TestCase):
    """load_config 文件不存在测试。"""

    def test_returns_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.json"
            with patch("gui.configs.config.config_path", return_value=fake_path):
                config = load_config()
                self.assertEqual(config, _DEFAULT_STATE)

    def test_returns_shallow_copy(self):
        """load_config 返回浅拷贝：顶层独立，嵌套字典共享引用（源代码行为）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.json"
            with patch("gui.configs.config.config_path", return_value=fake_path):
                config1 = load_config()
                config2 = load_config()
                # 顶层是独立的 dict 对象
                self.assertIsNot(config1, config2)
                # 但嵌套的 gui 字典是共享引用（dict() 浅拷贝）
                self.assertIs(config1["gui"], config2["gui"])

    def test_returned_dict_mutable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.json"
            with patch("gui.configs.config.config_path", return_value=fake_path):
                config = load_config()
                config["new_key"] = "value"
                self.assertIn("new_key", config)


class TestLoadConfigValidFile(unittest.TestCase):
    """load_config 有效文件测试。"""

    def test_returns_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {"gui": {"theme": "dark"}, "train": {"sr": 48000}, "models": ["a.pth"]}
            config_path.write_text(json.dumps(data), encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                self.assertEqual(config, data)

    def test_preserves_nested_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {
                "gui": {"window": {"x": 100, "y": 200}, "theme": "light"},
                "train": {"sr": 40000, "batch_size": 8},
                "models": [{"path": "/a.pth", "name": "model A"}],
            }
            config_path.write_text(json.dumps(data), encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                self.assertEqual(config["gui"]["window"]["x"], 100)
                self.assertEqual(config["train"]["batch_size"], 8)
                self.assertEqual(config["models"][0]["name"], "model A")

    def test_unicode_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {"gui": {"title": "语音转换工具"}, "train": {}, "models": []}
            config_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                self.assertEqual(config["gui"]["title"], "语音转换工具")

    def test_empty_dict_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                self.assertEqual(config, {})


class TestLoadConfigCorruptedFile(unittest.TestCase):
    """load_config 坏 JSON 文件测试。"""

    def test_returns_default_on_corrupted_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{invalid json!!!", encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                self.assertEqual(config, _DEFAULT_STATE)

    def test_creates_backup_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{bad json", encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                load_config()
                backup_path = config_path.with_suffix(config_path.suffix + ".bak")
                self.assertTrue(backup_path.exists())

    def test_backup_contains_original_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            original = "{this is bad json content"
            config_path.write_text(original, encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                load_config()
                backup_path = config_path.with_suffix(config_path.suffix + ".bak")
                self.assertEqual(backup_path.read_text(encoding="utf-8"), original)

    def test_original_file_removed_after_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{bad", encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                load_config()
                self.assertFalse(config_path.exists())

    def test_empty_file_corrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("", encoding="utf-8")
            with patch("gui.configs.config.config_path", return_value=config_path):
                config = load_config()
                # 空文件不是有效 JSON，应该返回默认
                self.assertEqual(config, _DEFAULT_STATE)


class TestSaveConfig(unittest.TestCase):
    """save_config 测试。"""

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.json"
            data = {"gui": {}, "train": {}, "models": []}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                self.assertTrue(config_path.exists())

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "a" / "b" / "c" / "config.json"
            data = {"gui": {}, "train": {}, "models": []}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                self.assertTrue(config_path.parent.exists())

    def test_writes_correct_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {"gui": {"theme": "dark"}, "train": {"sr": 48000}, "models": ["a.pth"]}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded, data)

    def test_unicode_content_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {"gui": {"title": "语音转换"}, "train": {}, "models": []}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["gui"]["title"], "语音转换")

    def test_atomic_write_no_tmp_left(self):
        """保存后不留下临时文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {"gui": {}, "train": {}, "models": []}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
                self.assertFalse(tmp_path.exists())

    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("old content", encoding="utf-8")
            data = {"gui": {"new": True}, "train": {}, "models": []}
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded, data)

    def test_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config({})
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded, {})

    def test_large_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {
                "gui": {f"key{i}": f"value{i}" for i in range(100)},
                "train": {f"param{i}": i for i in range(50)},
                "models": [{"path": f"/model{i}.pth", "name": f"Model {i}"} for i in range(20)],
            }
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(data)
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded, data)


class TestSaveLoadRoundtrip(unittest.TestCase):
    """保存-加载往返测试。"""

    def test_roundtrip_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            original = {
                "gui": {"theme": "dark", "window_size": [800, 600]},
                "train": {"sr": 48000, "batch_size": 16, "epochs": 100},
                "models": [
                    {"path": "/a.pth", "name": "Model A", "sr": 48000},
                    {"path": "/b.pth", "name": "Model B", "sr": 40000},
                ],
            }
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(original)
                loaded = load_config()
                self.assertEqual(loaded, original)

    def test_roundtrip_empty_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("gui.configs.config.config_path", return_value=config_path):
                save_config(dict(_DEFAULT_STATE))
                loaded = load_config()
                self.assertEqual(loaded, _DEFAULT_STATE)

    def test_multiple_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("gui.configs.config.config_path", return_value=config_path):
                for i in range(5):
                    data = {"gui": {"counter": i}, "train": {}, "models": []}
                    save_config(data)
                    loaded = load_config()
                    self.assertEqual(loaded["gui"]["counter"], i)


if __name__ == "__main__":
    unittest.main()
