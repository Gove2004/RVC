"""训练工具 Tab — 聚合入口（模型合并 / 模型信息 / 修正真名 三个独立模块）"""
from PySide6.QtWidgets import QWidget, QVBoxLayout

from gui.train.tabs import tools_merge, tools_inspect, tools_fixinfo


def build_tools_tab(win) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    layout.addWidget(tools_merge.build_group(win))
    layout.addWidget(tools_inspect.build_group(win))
    layout.addWidget(tools_fixinfo.build_group(win))
    layout.addStretch(1)
    return widget
