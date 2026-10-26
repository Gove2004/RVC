"""rvc.io.atomic — 原子写的两条契约：中断不留截断文件、Windows 占用可重试。

为什么单独测这个：训练产物按「文件存在即有效」消费，一个截断的 .pth 会被
下一轮静默复用；而 Windows 上 os.replace 撞到占用（杀毒/索引器）时的失败
在 Linux 开发机上是不可复现的，只能靠这里用 mock 锁住行为。
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rvc.io import atomic


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_write_creates_file_and_leaves_no_tmp(self):
        target = self.dir / "a.bin"
        atomic.atomic_write(target, lambda f: f.write(b"hello"))
        self.assertEqual(target.read_bytes(), b"hello")
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_writer_exception_leaves_old_content_intact(self):
        target = self.dir / "a.bin"
        target.write_bytes(b"old")

        def boom(f):
            f.write(b"partial")
            raise RuntimeError("序列化中途失败")

        with self.assertRaises(RuntimeError):
            atomic.atomic_write(target, boom)
        # 目标文件仍是旧内容——这是「不留截断文件」的核心保证
        self.assertEqual(target.read_bytes(), b"old")

    def test_permission_error_is_retried_then_succeeds(self):
        """Windows 上目标被短暂占用：前两次抛 PermissionError，第三次成功。"""
        target = self.dir / "a.bin"
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "被占用")
            return real_replace(src, dst)

        with mock.patch.object(atomic.os, "replace", side_effect=flaky):
            atomic.atomic_write(target, lambda f: f.write(b"ok"))

        self.assertEqual(calls["n"], 3)
        self.assertEqual(target.read_bytes(), b"ok")

    def test_permission_error_gives_up_after_attempts(self):
        target = self.dir / "a.bin"
        target.write_bytes(b"old")

        with mock.patch.object(atomic.os, "replace", side_effect=PermissionError(13, "被占用")):
            with mock.patch.object(atomic.time, "sleep") as sleeper:
                with self.assertRaises(PermissionError):
                    atomic.atomic_write(target, lambda f: f.write(b"new"))

        self.assertEqual(sleeper.call_count, atomic._REPLACE_ATTEMPTS - 1)
        self.assertEqual(target.read_bytes(), b"old")

    def test_other_oserror_is_not_retried(self):
        """ENOSPC/EROFS 之类的真实错误必须立刻抛出，不能靠重试掩盖。"""
        target = self.dir / "a.bin"
        with mock.patch.object(atomic.os, "replace", side_effect=OSError(28, "No space left")) as rep:
            with self.assertRaises(OSError):
                atomic.atomic_write(target, lambda f: f.write(b"x"))
        self.assertEqual(rep.call_count, 1)

    def test_text_round_trip(self):
        target = self.dir / "a.json"
        atomic.atomic_write_text(target, '{"a": 1}')
        self.assertEqual(target.read_text(encoding="utf-8"), '{"a": 1}')


class TestLoadJsonOrDefault(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmpdir.name)
        self.log = mock.Mock()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_file_returns_default(self):
        got = atomic.load_json_or_default(self.dir / "nope.json", {"d": 1}, logger=self.log)
        self.assertEqual(got, {"d": 1})

    def test_corrupt_file_is_backed_up_and_default_returned(self):
        target = self.dir / "bad.json"
        target.write_text("{not json", encoding="utf-8")
        got = atomic.load_json_or_default(target, {"d": 2}, logger=self.log)
        self.assertEqual(got, {"d": 2})
        # 用户数据可手工找回：备份文件必须存在且保留原内容
        self.assertEqual(target.with_suffix(".json.bak").read_text(encoding="utf-8"), "{not json")
        self.log.warning.assert_called()

    def test_valid_file_is_parsed(self):
        target = self.dir / "ok.json"
        target.write_text('{"k": "v"}', encoding="utf-8")
        self.assertEqual(atomic.load_json_or_default(target, {}, logger=self.log), {"k": "v"})


if __name__ == "__main__":
    unittest.main()
