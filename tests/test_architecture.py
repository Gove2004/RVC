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
LOWER_ONLY_LAYERS = {
    "core": UPPER_LAYERS,
    "runtime": UPPER_LAYERS,
    "dsp": UPPER_LAYERS,  # S3 建包后立即生效
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
        """core/runtime/dsp 不得 import 任何上层包。"""
        violations = []
        for layer, forbidden in LOWER_ONLY_LAYERS.items():
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
        forbidden = ["rvc.runtime.cuda_graph", "rvc.pipeline.cuda_graph"]
        violations = []
        for module in RVC.rglob("*.py"):
            text = module.read_text(encoding="utf-8")
            for name in forbidden:
                if name in text:
                    rel = module.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel} references {name}")
        assert not violations, "已删模块被引用:\n" + "\n".join(violations)


if __name__ == "__main__":
    unittest.main()
