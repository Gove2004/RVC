"""pytest 配置 — 确保项目根目录在 sys.path 中。"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，这样测试可以 import rvc / gui
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
