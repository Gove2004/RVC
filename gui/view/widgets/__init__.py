"""推理 GUI 通用组件包 — 按职责分模块，符号在此聚合再导出。

调用方统一 `from gui.view.widgets import X`；子模块按职责划分：
- pickers：路径选择按钮、中间省略标签
- sliders：浮点滑杆、滑杆行辅助、双滑块范围选择器
- mode_toggle：三档运行态切换
- loading：模型加载后台线程
"""
from gui.view.widgets.pickers import ElidedLabel, PathPickerButton
from gui.view.widgets.sliders import DoubleSlider, RangeSlider, create_slider_row
from gui.view.widgets.mode_toggle import ModeToggle
from gui.view.widgets.loading import LoadThread

__all__ = [
    "LoadThread", "create_slider_row", "DoubleSlider", "RangeSlider",
    "PathPickerButton", "ElidedLabel", "ModeToggle",
]
