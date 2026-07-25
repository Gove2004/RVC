"""UI 组件工具 — 共享的 widgets 和分隔线等"""

from PySide6.QtWidgets import QFrame
from gui.styles.colors import Colors


def sep() -> QFrame:
    """水平分隔线"""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{Colors.DIVIDER}")
    return f
