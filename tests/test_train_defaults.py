"""训练向导存档（rvc/train/cli/defaults.py）的读写契约。

这个模块曾经长期零测试覆盖，代价是 `_load_raw` 里漏了一个
`from rvc.io.atomic import load_json_or_default` 的导入——`load_defaults()`
每次调用都抛 NameError，训练向导的「上次答案」预填完全不可用，而 189 个测试
全绿。这个文件就是补上那条覆盖。

重点覆盖三件事：
1. `load_defaults()` 在存档缺失/损坏/正常三种情况下都能返回完整键集（不抛）
2. 存档里的非法值不许变成默认值（白名单校验）
3. `save_defaults` → `load_defaults` 往返一致
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rvc.core.config import HUBERT_DEFAULT
from rvc.train.cli import DEFAULTS
from rvc.train.cli import defaults as defaults_mod
from rvc.train.cli.defaults import PERSISTED_KEYS, _sanitize, load_defaults, save_defaults


class _TempStore:
    """把存档路径临时指到 tmp 目录，避免碰到用户真实存档。"""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "train_cli_defaults.json"
        self._patch = mock.patch.object(defaults_mod, "TRAIN_CLI_DEFAULTS_FILE", self.path)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()
        return False

    def write(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestLoadDefaults(unittest.TestCase):
    # 返回值必须覆盖向导要问的每一项，否则 ask() 会拿到 KeyError
    REQUIRED_KEYS = set(PERSISTED_KEYS)

    def test_missing_file_returns_builtin_defaults(self):
        with _TempStore() as store:
            self.assertFalse(store.path.exists())
            d = load_defaults()
        self.assertTrue(self.REQUIRED_KEYS <= set(d))
        self.assertEqual(d["exp_name"], DEFAULTS["exp"])
        self.assertEqual(d["sr"], DEFAULTS["sr"])
        self.assertIsNone(d["batch_size"])  # None = 按显存现算

    def test_default_variant_is_hubert_default(self):
        """存档里没有 hubert 时必须落到 HUBERT_DEFAULT（base），不是写死字面量。"""
        with _TempStore():
            d = load_defaults()
        self.assertEqual(d["hubert"], HUBERT_DEFAULT)
        self.assertEqual(HUBERT_DEFAULT, "base")

    def test_corrupt_file_falls_back_and_backs_up(self):
        with _TempStore() as store:
            store.path.write_text("{ 这不是 json", encoding="utf-8")
            d = load_defaults()
            bak = store.path.with_suffix(store.path.suffix + ".bak")
            self.assertTrue(bak.exists(), "损坏的存档必须备份为 .bak，不能静默丢弃")
        self.assertTrue(self.REQUIRED_KEYS <= set(d))

    def test_saved_values_win_over_builtin(self):
        with _TempStore() as store:
            store.write({"exp_name": "mymodel", "sr": "48k", "hubert": "chinese",
                         "epochs": 33, "batch_size": 4})
            d = load_defaults()
        self.assertEqual(d["exp_name"], "mymodel")
        self.assertEqual(d["sr"], "48k")
        self.assertEqual(d["hubert"], "chinese")
        self.assertEqual(d["epochs"], 33)
        self.assertEqual(d["batch_size"], 4)

    def test_nonexistent_dir_is_dropped(self):
        """存档里的目录在本机不存在时丢弃，避免默认值指向建不出来的盘。"""
        with _TempStore() as store:
            store.write({"input_dir": r"Z:\definitely\not\here"})
            d = load_defaults()
        self.assertEqual(d["input_dir"], "")


class TestSanitize(unittest.TestCase):
    def test_unknown_hubert_rejected(self):
        self.assertNotIn("hubert", _sanitize({"hubert": "japanese"}))
        self.assertNotIn("hubert", _sanitize({"hubert": ""}))
        self.assertEqual(_sanitize({"hubert": "base"})["hubert"], "base")
        self.assertEqual(_sanitize({"hubert": "chinese"})["hubert"], "chinese")

    def test_unknown_sr_rejected(self):
        self.assertNotIn("sr", _sanitize({"sr": "44k"}))
        self.assertEqual(_sanitize({"sr": "40k"})["sr"], "40k")

    def test_out_of_range_lr_rejected(self):
        self.assertNotIn("lr", _sanitize({"lr": 0}))
        self.assertNotIn("lr", _sanitize({"lr": -1e-4}))
        self.assertNotIn("lr", _sanitize({"lr": 0.5}))
        self.assertNotIn("lr", _sanitize({"lr": True}))  # bool 不是数值
        self.assertEqual(_sanitize({"lr": 1e-4})["lr"], 1e-4)

    def test_non_positive_counts_rejected(self):
        for key in ("epochs", "save_every", "keep_ckpts"):
            self.assertNotIn(key, _sanitize({key: 0}))
            self.assertNotIn(key, _sanitize({key: True}))
            self.assertEqual(_sanitize({key: 3})[key], 3)

    def test_batch_size_rejected_when_below_one(self):
        self.assertNotIn("batch_size", _sanitize({"batch_size": 0}))
        self.assertEqual(_sanitize({"batch_size": 2})["batch_size"], 2)

    def test_blank_exp_name_rejected(self):
        self.assertNotIn("exp_name", _sanitize({"exp_name": "   "}))

    def test_unknown_keys_dropped(self):
        self.assertEqual(_sanitize({"nonsense": 1}), {})


class TestRoundtrip(unittest.TestCase):
    def test_save_then_load(self):
        cfg = {
            "input_dir": "", "output_root": "/tmp", "exp_name": "rt", "sr": "40k",
            "hubert": "base", "batch_size": 8, "lr": 2e-4, "epochs": 12,
            "save_every": 2, "keep_ckpts": 3, "use_pretrained": False,
            "per": 3.0,  # 非持久化键，应被 save_defaults 丢弃
        }
        with _TempStore() as store:
            save_defaults(cfg)
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            d = load_defaults()
        self.assertNotIn("per", raw)
        self.assertEqual(set(raw) <= set(PERSISTED_KEYS), True)
        for key in ("exp_name", "sr", "hubert", "batch_size", "lr", "epochs"):
            self.assertEqual(d[key], cfg[key], key)

    def test_save_is_atomic_no_tmp_left(self):
        with _TempStore() as store:
            save_defaults({"exp_name": "a", "sr": "40k"})
            leftovers = list(store.path.parent.glob("*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
