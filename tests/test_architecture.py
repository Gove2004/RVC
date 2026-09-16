"""架构守护测试 — 静态检查 import 依赖方向，禁止反向依赖。

分层规则（docs/rewrite/03-目标架构设计.md §1.2）随重写步骤逐步收紧：
- S2 起：core/runtime 不得依赖任何上层（pipeline/streaming/train/models/io/gui/autodl）；
- S4 起：models 不得依赖 pipeline/streaming/train；
- S6 起：pipeline 不得依赖 streaming。
"""
import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RVC = PROJECT_ROOT / "rvc"

UPPER_LAYERS = {"pipeline", "streaming", "train", "models", "io", "gui", "autodl"}

# 已删除的旧模块，任何 rvc 代码不得再引用（防止引用腐化回潮）
DEAD_MODULES = [
    "rvc.runtime.cuda_graph",
    "rvc.pipeline.cuda_graph",
    "rvc.streaming.mix",
    "rvc.streaming.alignment",
    "rvc.train.mel",
    "rvc.pipeline.registry",
]

# 分层规则：layer -> 禁止 import 的包
FORBIDDEN_BY_LAYER = {
    "core": UPPER_LAYERS,
    "runtime": UPPER_LAYERS,
    "dsp": UPPER_LAYERS,  # S3 建包后立即生效
    "models": UPPER_LAYERS - {"models"},  # S4 起：models 不得依赖 pipeline/streaming/train/io
    "pipeline": {"streaming", "gui", "autodl"},  # S6 起：pipeline 不得依赖 streaming（D2）
}


def _local_imports(module: Path):
    """返回模块内 import 的 rvc 顶层子包名集合。"""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "rvc" and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue
            parts = node.module.split(".")
            if parts[0] == "rvc" and len(parts) > 1:
                found.add(parts[1])
    return found


class TestImportDirection(unittest.TestCase):
    def test_lower_layers_import_no_upper_layers(self):
        """各低层不得 import 被禁的上层包（规则见 FORBIDDEN_BY_LAYER）。"""
        violations = []
        for layer, forbidden in FORBIDDEN_BY_LAYER.items():
            layer_dir = RVC / layer
            if not layer_dir.is_dir():
                continue
            for module in layer_dir.rglob("*.py"):
                bad = _local_imports(module) & forbidden
                if bad:
                    rel = module.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel} -> {sorted(bad)}")
        assert not violations, "依赖方向违规:\n" + "\n".join(violations)

    def test_deleted_modules_not_referenced(self):
        """已删除的旧模块不得再被引用（防止引用腐化回潮）。"""
        import re
        violations = []
        for module in RVC.rglob("*.py"):
            text = module.read_text(encoding="utf-8")
            for name in DEAD_MODULES:
                # 词边界匹配：rvc.train.mel 不误伤 rvc.train.mel_processing
                if re.search(re.escape(name) + r"(?![A-Za-z0-9_])", text):
                    rel = module.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel} references {name}")
        assert not violations, "已删模块被引用:\n" + "\n".join(violations)

    def test_renamed_symbols_not_referenced(self):
        """已改名符号不得在代码中残留（S7：RealtimeEngine → VoiceEngine）。"""
        import re
        violations = []
        for module in list(RVC.rglob("*.py")) + list((PROJECT_ROOT / "gui").rglob("*.py")):
            text = module.read_text(encoding="utf-8")
            if re.search(r"\bRealtimeEngine\b", text):
                rel = module.relative_to(PROJECT_ROOT)
                violations.append(str(rel))
        assert not violations, "旧门面名残留:\n" + "\n".join(violations)

    def test_gui_view_no_rvc_imports(self):
        """S10/D5：gui 的 View 层禁止 import rvc.*（唯一例外：TYPE_CHECKING 类型注解）。

        顶层 import（ast.Import / ImportFrom 在模块体直接出现）一律违规；
        if TYPE_CHECKING: 块内的 import 属于 If 节点，不算顶层，放行。
        """
        violations = []
        for view_dir in (PROJECT_ROOT / "gui" / "infer" / "view",
                         PROJECT_ROOT / "gui" / "train" / "view"):
            for module in view_dir.rglob("*.py"):
                tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
                found = set()
                for node in tree.body:
                    nodes = [node]
                    if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
                        # TYPE_CHECKING 守卫的块放行（类型注解例外）
                        continue
                    while nodes:
                        n = nodes.pop()
                        if isinstance(n, ast.Import):
                            for alias in n.names:
                                if alias.name.split(".")[0] == "rvc":
                                    found.add(alias.name)
                        elif isinstance(n, ast.ImportFrom):
                            if n.level == 0 and n.module and n.module.split(".")[0] == "rvc":
                                found.add(n.module)
                        nodes.extend(ast.iter_child_nodes(n))
                if found:
                    rel = module.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel} -> {sorted(found)}")
        assert not violations, "View 层 import rvc.* 违规:\n" + "\n".join(violations)


def _is_type_checking_guard(test) -> bool:
    """识别 `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` 守卫。"""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


if __name__ == "__main__":
    unittest.main()
