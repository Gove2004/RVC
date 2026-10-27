"""回归守卫：全仓不许出现「函数体引用了模块中根本没绑定的全局名」。

来历：`rvc/train/cli/defaults.py` 用了 `load_json_or_default` 却没 import，
`gui/configs/config.py` 用了 `json.dumps` 却没 `import json`。两处都是**调用必抛
NameError**、import 阶段却毫无症状，其中一个还被退出路径的 try/except 吞掉，
表现为「设置静默全丢」。当时 200 多个测试全绿，因为这两条分支零覆盖。

用标准库 `symtable` 做静态判定：函数作用域里被引用、解析到全局、而模块级既没绑定
也不是内建名的，就是未定义。当前基线是 0 处，新增一处的成本远低于线上炸一次。
"""
import builtins
import symtable
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCAN_DIRS = ("rvc", "gui")

# 内建名 + 模块级由导入机制注入的 dunder
_OK_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__", "__all__",
    "__annotations__", "__dict__", "__class__",
}


def _module_bindings(table: symtable.SymbolTable) -> set[str]:
    """模块级所有被绑定的名字（import / def / class / 赋值 / 解包 / 循环变量…）。"""
    names = set()
    for sym in table.get_symbols():
        if (sym.is_assigned() or sym.is_imported() or sym.is_namespace()
                or sym.is_parameter()):
            names.add(sym.get_name())
    return names


def _walk(table, path, mod_names, out, stack=()):
    where = stack + (table.get_name(),)
    for sym in table.get_symbols():
        name = sym.get_name()
        if not sym.is_referenced():
            continue
        if sym.is_local() or sym.is_parameter() or sym.is_free() or sym.is_imported():
            continue
        if name in _OK_NAMES or name in mod_names:
            continue
        # 函数体内 `global X` 且自己也赋值：由该函数动态创建，不是缺陷
        if sym.is_assigned():
            continue
        out.append((path, ".".join(where), name))
    for child in table.get_children():
        _walk(child, path, mod_names, out, where)


def scan_undefined_names(root: Path) -> list[tuple[str, str, str]]:
    problems = []
    for pkg in _SCAN_DIRS:
        for f in sorted((root / pkg).rglob("*.py")):
            if "__pycache__" in f.parts or ".venv" in f.parts:
                continue
            src = f.read_text(encoding="utf-8")
            top = symtable.symtable(src, str(f), "exec")
            _walk(top, str(f.relative_to(root)), _module_bindings(top), problems)
    return problems


class TestNoUndefinedGlobalNames(unittest.TestCase):
    def test_no_undefined_names(self):
        problems = scan_undefined_names(_PROJECT_ROOT)
        if problems:
            detail = "\n".join(f"  {p} [{scope}] {name}" for p, scope, name in problems)
            self.fail(
                "以下位置引用了模块中未绑定的全局名（调用时必抛 NameError）：\n" + detail
            )

    def test_scanner_catches_missing_import(self):
        """自检：扫描器真的能抓到漏 import，而不是永远返回空。"""
        import tempfile

        src = "def f():\n    return json.dumps({})\n"
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "rvc"
            pkg.mkdir()
            (pkg / "broken.py").write_text(src, encoding="utf-8")
            problems = scan_undefined_names(Path(d))
        self.assertTrue(
            any(name == "json" for _, _, name in problems),
            f"扫描器未能识别缺失的 import，返回 {problems}",
        )


# ── 死引用守卫：防"删掉功能留下指向不存在属性的引用" ──

def _has_attr_or_submodule(mod, mod_name: str, attr: str) -> bool:
    """attr 存在于 mod，或 attr 是 mod 的子模块（import 子模块会挂成包属性）。"""
    import importlib

    if hasattr(mod, attr):
        return True
    try:
        importlib.import_module(f"{mod_name}.{attr}")
        return True
    except Exception:
        return False


def scan_dead_references(root: Path) -> list[tuple[str, int, str]]:
    """找 `from X import Y` / `X.attr` 里 Y、attr 在 X 中已不存在的情况。

    只覆盖可静态判定的，不做类型推断。相对导入跳过（由测试运行时兜底）。
    """
    import ast
    import importlib
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    problems = []
    for pkg in _SCAN_DIRS:
        for f in sorted((root / pkg).rglob("*.py")):
            if "__pycache__" in f.parts or ".venv" in f.parts:
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            rel = str(f.relative_to(root))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level or not node.module:
                        continue
                    if not node.module.startswith(("rvc", "gui")):
                        continue
                    try:
                        mod = importlib.import_module(node.module)
                    except Exception:
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if not _has_attr_or_submodule(mod, node.module, alias.name):
                            problems.append((
                                rel, node.lineno,
                                f"from {node.module} import {alias.name}"))
    return problems


class TestNoDeadReferences(unittest.TestCase):
    """删掉一个功能后，不许留下指向已不存在属性的引用。

    来历：目标 4 的真实感链整条删除后，`params_tab.py` 的两处 docstring 仍在
    描述"真实感"，MEMORY 也仍以现在时把 `process` 映射到 `runner.realism`
    （该属性已不存在，照它重建剖析脚本会直接 AttributeError）。
    这类残留不会让测试变红，只会误导后来的人，所以在静态层扫掉。
    """

    def test_no_dead_references(self):
        problems = scan_dead_references(_PROJECT_ROOT)
        if problems:
            detail = "\n".join(f"  {p}:{ln} {msg}" for p, ln, msg in problems)
            self.fail("以下位置引用了已不存在的属性（删功能后的残留）：\n" + detail)

    def test_scanner_catches_dead_reference(self):
        """自检：扫描器能抓到指向不存在属性的 import。"""
        import tempfile

        src = "from rvc.core.config import NO_SUCH_THING\n"
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "rvc"
            pkg.mkdir()
            (pkg / "dead.py").write_text(src, encoding="utf-8")
            problems = scan_dead_references(Path(d))
        self.assertTrue(
            any("NO_SUCH_THING" in msg for _, _, msg in problems),
            f"扫描器未能识别死引用，返回 {problems}",
        )


if __name__ == "__main__":
    unittest.main()
