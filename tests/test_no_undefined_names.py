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


if __name__ == "__main__":
    unittest.main()
