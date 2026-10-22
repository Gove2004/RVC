"""分层契约的自动化守卫 — README 的「八层单向、无环」不能只靠评审。

三条硬规则（都是历史上真被破坏过的方向）：
1. `rvc/**` 不得 import `gui`（引擎必须能脱离 GUI 独立使用）
2. `gui/view/**` 不得在运行时 import `rvc`（只能 TYPE_CHECKING 引用类型；
   View 经 Controller 取数，否则引擎/视图又会缠在一起）
3. 包级依赖必须单向且无环，层级序由 RANKS 表达

为什么用 AST 而不是 import：规则要在没有 GPU/torch 的环境里也能跑，
且 `TYPE_CHECKING` 里的导入运行时根本不会发生，必须静态判断。
"""
import ast
import pathlib
import unittest
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 层级序：数字小的是下层。依赖只允许指向严格更小的数字。
RANKS = {
    "rvc.core": 0,
    "rvc.runtime": 1,
    "rvc.dsp": 2,
    "rvc.nn": 2,
    "rvc.io": 2,
    "rvc.models": 3,
    "rvc.pipeline": 4,
    "rvc.streaming": 5,
    "rvc.train": 5,
    "gui": 6,
}


def _package_of(path: pathlib.Path) -> str:
    """依赖图节点粒度：rvc 到二级目录（rvc.pipeline 等），gui 整体算一层。"""
    parts = path.relative_to(ROOT).parts
    if parts[0] == "rvc":
        return ".".join(parts[:2]) if len(parts) > 2 else "rvc"
    return parts[0]


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _runtime_imports(tree: ast.Module) -> list[tuple[str, bool]]:
    """返回 [(被导入模块全名, 是否在 TYPE_CHECKING 块内)]。"""
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, type_checking: bool) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            for child in node.body:
                visit(child, True)
            return
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, type_checking))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, type_checking))
        for child in ast.iter_child_nodes(node):
            visit(child, type_checking)

    visit(tree, False)
    return found


def _parse_all() -> dict[pathlib.Path, ast.Module]:
    trees = {}
    for folder in ("rvc", "gui"):
        for path in (ROOT / folder).rglob("*.py"):
            trees[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return trees


class TestLayering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = _parse_all()

    def test_rvc_never_imports_gui(self):
        offenders = []
        for path, tree in self.trees.items():
            if path.relative_to(ROOT).parts[0] == "gui":
                continue
            for module, _tc in _runtime_imports(tree):
                if module == "gui" or module.startswith("gui."):
                    offenders.append(f"{path.relative_to(ROOT)} → {module}")
        self.assertEqual(offenders, [], "rvc 不得依赖 gui（引擎要能独立使用）")

    def test_view_layer_does_not_import_rvc_at_runtime(self):
        offenders = []
        for path, tree in self.trees.items():
            if not str(path.relative_to(ROOT)).replace("\\", "/").startswith("gui/view/"):
                continue
            for module, type_checking in _runtime_imports(tree):
                if module == "rvc" or module.startswith("rvc."):
                    if not type_checking:
                        offenders.append(f"{path.relative_to(ROOT)} → {module}")
        self.assertEqual(offenders, [],
                         "View 层只能 TYPE_CHECKING 引用 rvc 类型；运行时取数走 Controller")

    def test_package_graph_is_acyclic_and_downward(self):
        graph = defaultdict(set)
        for path, tree in self.trees.items():
            src = _package_of(path)
            for module, _tc in _runtime_imports(tree):
                top = module.split(".")[0]
                if top not in ("rvc", "gui"):
                    continue
                dst = ".".join(module.split(".")[:2]) if top == "rvc" else "gui"
                if dst != src:
                    graph[src].add(dst)

        problems = []
        for src, targets in graph.items():
            src_rank = RANKS[src]
            for dst in targets:
                if dst not in RANKS:
                    problems.append(f"{src} → {dst}（未登记层级）")
                elif RANKS[dst] >= src_rank:
                    problems.append(f"{src}(rank {src_rank}) → {dst}(rank {RANKS[dst]})")

        # 环检测（rank 规则通过时理论上不可能有环，但仍显式验证一次，
        # 免得以后改 RANKS 时把环放进来还以为规则生效）
        state: dict[str, int] = {}

        def dfs(node: str, stack: list[str]) -> None:
            state[node] = 1
            for nxt in graph.get(node, ()):
                if state.get(nxt) == 1:
                    problems.append("环: " + " → ".join(stack + [node, nxt]))
                elif state.get(nxt) is None:
                    dfs(nxt, stack + [node])
            state[node] = 2

        for node in list(graph):
            if state.get(node) is None:
                dfs(node, [])

        self.assertEqual(problems, [], "包级依赖必须单向、无环、且与 README 的层级一致")

    def test_rank_table_covers_every_package(self):
        seen = {_package_of(p) for p in self.trees}
        missing = sorted(p for p in seen if p not in RANKS and p != "rvc")
        self.assertEqual(missing, [], "新增包必须先在 RANKS 里登记层级")


if __name__ == "__main__":
    unittest.main()
