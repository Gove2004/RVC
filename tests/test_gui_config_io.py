"""GUI 存档读写契约（gui/configs/config.py）。

这个模块曾经 `save_config` 里用了 `json.dumps` 却没 `import json`——每次保存都抛
NameError，而 `_tray_quit` 把它包在 try/except 里只记一行日志，于是**设置静默全丢**：
软件照常退出，用户以为存住了，下次启动全是旧值。这类缺陷 import 阶段发现不了，
只能在真正调用时炸，所以这里坚持走真实写盘（临时目录），不用 mock。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui.configs import config as config_mod
from gui.configs.config import load_config, save_config


class _TempState:
    """把 STATE_FILE 临时指到 tmp 目录，不碰用户真实存档。"""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "save_state.json"
        self._patch = mock.patch.object(config_mod, "STATE_FILE", self.path)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()
        return False


class TestSaveConfig(unittest.TestCase):
    def test_save_then_load_roundtrip(self):
        state = {"gui": {"volume": 1.5, "buffer": {"block_time": 0.1}}, "models": []}
        with _TempState() as store:
            save_config(state)
            self.assertTrue(store.path.exists(), "save_config 必须真的写出文件")
            loaded = load_config()
        self.assertEqual(loaded, state)

    def test_written_file_is_valid_utf8_json(self):
        with _TempState() as store:
            save_config({"gui": {"名字": "中文"}})
            raw = store.path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw)["gui"]["名字"], "中文")

    def test_missing_file_returns_default_gui_key(self):
        with _TempState():
            loaded = load_config()
        self.assertEqual(loaded, {"gui": {}})

    def test_corrupt_file_falls_back_and_backs_up(self):
        with _TempState() as store:
            store.path.write_text("{ 坏掉的", encoding="utf-8")
            loaded = load_config()
            self.assertTrue(
                store.path.with_suffix(store.path.suffix + ".bak").exists(),
                "损坏的存档必须备份，不能静默丢弃用户数据",
            )
        self.assertEqual(loaded, {"gui": {}})

    def test_overwrite_replaces_previous_content(self):
        with _TempState() as store:
            save_config({"gui": {"a": 1}, "models": [1, 2]})
            save_config({"gui": {"b": 2}})
            loaded = load_config()
            self.assertNotIn("models", loaded)
            self.assertEqual(loaded["gui"], {"b": 2})


if __name__ == "__main__":
    unittest.main()
