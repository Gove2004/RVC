"""rvc.io.atomic — 原子写的两条契约：中断不留截断文件、Windows 占用可重试。

为什么单独测这个：训练产物按「文件存在即有效」消费，一个截断的 .pth 会被
下一轮静默复用；而 Windows 上 os.replace 撞到占用（杀毒/索引器）时的失败
在 Linux 开发机上是不可复现的，只能靠这里用 mock 锁住行为。
"""
import builtins
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rvc.io import atomic

# 并发写用例的数组长度（够大以拉长写窗口，又不拖慢单测）
_CONCURRENT_SIZE = 200_000


def _write_full_array(path: str, idx: int) -> None:
    """多进程写者（必须是模块级：spawn 需要可 pickle）。"""
    from rvc.train.atomic_save import save_npy_atomic

    save_npy_atomic(path, np.full((_CONCURRENT_SIZE,), float(idx), dtype=np.float32))


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

    def test_tmp_name_is_unique_per_call(self):
        """tmp 名必须每次唯一：多进程并发写同一目标时不得共用 tmp。

        训练侧 DataLoader 的 worker 是独立进程，会并发首次填充同一批样本缓存
        （dataset._cached_wav/_cached_spec 惰性写盘）。固定 tmp 名会让先完成者
        把 tmp 移走、后到者在 os.replace 处抛 FileNotFoundError——而退避只重试
        PermissionError，兜不住它，表现为「新实验第一个 epoch 偶发崩溃」。
        """
        target = self.dir / "a.bin"
        names = []
        real_open = builtins.open

        def spy_open(file, *a, **kw):
            names.append(str(file))
            return real_open(file, *a, **kw)

        with mock.patch("builtins.open", side_effect=spy_open):
            atomic.atomic_write(target, lambda f: f.write(b"x"))
            atomic.atomic_write(target, lambda f: f.write(b"y"))

        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0], names[1], "两次写的 tmp 名相同，并发写会互撞")
        # 两次都不得残留
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_failed_write_leaves_no_tmp(self):
        """writer 抛错后必须清掉自己的 tmp，否则缓存目录会积累垃圾文件。"""
        target = self.dir / "a.bin"

        def boom(f):
            f.write(b"partial")
            raise RuntimeError("序列化中途失败")

        with self.assertRaises(RuntimeError):
            atomic.atomic_write(target, boom)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_concurrent_writers_do_not_collide(self):
        """多进程写同一目标：每个写者独占 tmp，最终文件是某个写者的完整内容。

        这是回归锁：固定 tmp 名下本用例约 20% 概率抛 FileNotFoundError。
        用真实多进程复刻 DataLoader worker 的并发写入（写函数必须是模块级，
        spawn 需要可 pickle）。
        """
        target = self.dir / "shared.npy"
        ctx = multiprocessing.get_context("spawn")
        for _ in range(6):
            procs = [ctx.Process(target=_write_full_array, args=(str(target), i))
                     for i in range(2)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            for p in procs:
                self.assertEqual(p.exitcode, 0, "并发写同一目标时进程异常退出")

        got = np.load(target)
        self.assertEqual(len(got), _CONCURRENT_SIZE)
        # 内容必须是某一个写者写下的完整数组，不能是两者交错
        self.assertEqual(len(set(np.unique(got).tolist())), 1)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])


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
