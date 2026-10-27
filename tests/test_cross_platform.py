"""守卫：跨平台（Windows/Linux）不变量 —— 目标 6「适配 Windows 和 Linux」。

来历：训练模块要求双平台可用，但项目在 Windows 上开发，Linux 专属的坑不会被
日常使用暴露。已系统审计过以下类别，逐项确认干净或有守卫：

1. Windows 专有 API（ctypes.windll / winreg / msvcrt / os.startfile）
2. 硬编码路径分隔符
3. subprocess shell=True
4. POSIX 专有信号（SIGKILL/SIGHUP/SIGUSR*）——Windows 上不存在
5. 文件锁（fcntl / msvcrt）
6. 多进程启动方式：**必须显式 spawn**。Linux 默认 fork，而 setup() 在创建
   DataLoader 前已把模型搬上 CUDA，fork 出的 worker 继承半初始化的 CUDA 上下文，
   表现为卡死或 "Cannot re-initialize CUDA in forked subprocess"。
   Windows 只有 spawn 所以看不出来，Linux 上必踩。

本测试把这些不变量的**关键几项**钉住，防止以后回退。
（不试图覆盖全部审计项：那种测试会因过度耦合实现细节而频繁误报。）
"""
import ast
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _guarded_by_early_return(src: str, marker: str) -> bool:
    """marker 所在函数里，是否在它之前就有 `if os.name != "nt": raise/return`。

    扫描器只认外层 if 块会漏掉"提前返回式守卫"，这里专门识别那种写法。
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_src = ast.get_source_segment(src, node) or ""
        if marker not in body_src:
            continue
        marker_line = next((i for i, ln in enumerate(lines, 1)
                            if marker in ln and node.lineno <= i <= node.end_lineno), None)
        if marker_line is None:
            continue
        for child in node.body:
            if not isinstance(child, ast.If):
                continue
            if child.lineno >= marker_line:
                continue
            test_src = ast.get_source_segment(src, child.test) or ""
            if "os.name" in test_src:
                # 该 if 体内是否 raise / return
                for st in child.body:
                    if isinstance(st, (ast.Raise, ast.Return)):
                        return True
    return False


class TestWindowsApiGuarded(unittest.TestCase):
    """Windows 专有 API 必须有平台守卫。"""

    def test_envcheck_memory_guarded(self):
        src = _read("rvc/train/cli/envcheck.py")
        self.assertIn('if os.name == "nt"', src)
        self.assertIn("GlobalMemoryStatusEx", src)
        # 必须有 POSIX 分支（os.sysconf）
        self.assertIn("sysconf", src)

    def test_recycle_guarded_by_early_return(self):
        """回收站用 ctypes.windll，但守卫是提前返回式，不是外层 if 块。"""
        src = _read("gui/controller/recycle.py")
        self.assertIn("ctypes.windll", src)
        self.assertTrue(
            _guarded_by_early_return(src, "ctypes.windll"),
            "recycle_file 在调用 ctypes.windll 前没有 os.name 守卫——"
            "Linux 上会 AttributeError 而非可读的 NotImplementedError",
        )


class TestCreationFlagsSafeOnPosix(unittest.TestCase):
    """CREATE_NO_WINDOW 在非 Windows 必须为 0。

    subprocess 的 creationflags 是 Windows 专有参数，CPython 只在
    `creationflags != 0` 时才抛 ValueError（POSIX 分支）。所以"非 Windows 置 0
    仍照传"是正确写法，不是 bug——但前提是值真的为 0。
    """

    def test_constant_is_zero_off_windows(self):
        src = _read("rvc/io/__init__.py")
        self.assertIn('if os.name == "nt" else 0', src,
                      "CREATE_NO_WINDOW 非 Windows 分支必须为 0，否则 Linux 上 "
                      "subprocess 会抛 ValueError（creationflags 仅 Windows 支持）")


class TestMultiprocessingUsesSpawn(unittest.TestCase):
    """DataLoader worker 必须显式 spawn（Linux 默认 fork + CUDA = 挂死）。"""

    def test_trainer_forces_spawn(self):
        src = _read("rvc/train/trainer.py")
        self.assertIn('multiprocessing_context"] = "spawn"', src,
                      "训练 DataLoader 不再显式指定 spawn——Linux 上 fork 的 worker "
                      "会继承父进程已初始化的 CUDA 上下文，导致卡死或 "
                      "'Cannot re-initialize CUDA in forked subprocess'")

    def test_persistent_workers_cleaned_up(self):
        """persistent_workers 不会在最后一个 epoch 自动退出，必须显式关闭。"""
        src = _read("rvc/train/trainer.py")
        self.assertIn("persistent_workers", src)
        # 关闭逻辑存在（_shutdown_workers 或类似）
        self.assertTrue(
            "_shutdown_workers" in src or "shutdown" in src.lower(),
            "persistent_workers 缺少显式关闭，训练结束后进程会被 worker 持有",
        )


class TestNoPosixOnlyOrPlatformHostileCalls(unittest.TestCase):
    """全仓不许出现 POSIX 专有信号、shell=True、POSIX 文件锁。"""

    _FORBIDDEN = ("SIGKILL", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGALRM",
                  "shell=True", "fcntl.", "msvcrt.locking", "os.startfile")

    def _scan(self, root: Path) -> list[str]:
        hits = []
        for pkg in ("rvc", "gui"):
            base = root / pkg
            if not base.is_dir():
                continue
            for f in sorted(base.rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    code = line.split("#", 1)[0]
                    for bad in self._FORBIDDEN:
                        if bad in code:
                            hits.append(f"{f.relative_to(root)}:{i}  {bad}")
        return hits

    def test_no_forbidden_calls(self):
        hits = self._scan(_PROJECT_ROOT)
        self.assertFalse(hits, "发现平台不友好调用：\n" + "\n".join(hits))

    def test_scanner_catches_forbidden_call(self):
        """自检：扫描器真的能抓到平台不友好调用，而不是永远返回空。"""
        import tempfile

        src = "import subprocess\nsubprocess.run('ls', shell=True)\n"
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "rvc"
            pkg.mkdir()
            (pkg / "bad.py").write_text(src, encoding="utf-8")
            hits = self._scan(Path(d))
        self.assertTrue(any("shell=True" in h for h in hits),
                        f"扫描器未能识别 shell=True，返回 {hits}")


class TestEarlyReturnGuardDetector(unittest.TestCase):
    """自检 `_guarded_by_early_return`：能识别真守卫、不误判无守卫。"""

    def test_detects_early_return_guard(self):
        src = (
            "import os\n"
            "def f():\n"
            "    if os.name != 'nt':\n"
            "        raise NotImplementedError('win only')\n"
            "    import ctypes\n"
            "    return ctypes.windll.shell32.Foo()\n"
        )
        self.assertTrue(_guarded_by_early_return(src, "ctypes.windll"))

    def test_rejects_unguarded(self):
        src = (
            "def f():\n"
            "    import ctypes\n"
            "    return ctypes.windll.shell32.Foo()\n"
        )
        self.assertFalse(_guarded_by_early_return(src, "ctypes.windll"))

    def test_guard_after_call_does_not_count(self):
        """守卫出现在调用之后无效。"""
        src = (
            "import os\n"
            "def f():\n"
            "    import ctypes\n"
            "    x = ctypes.windll.shell32.Foo()\n"
            "    if os.name != 'nt':\n"
            "        raise NotImplementedError('win only')\n"
            "    return x\n"
        )
        self.assertFalse(_guarded_by_early_return(src, "ctypes.windll"))


if __name__ == "__main__":
    unittest.main()
