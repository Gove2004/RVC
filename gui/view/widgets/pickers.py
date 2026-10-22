"""路径与文本展示控件 — 文件选择按钮、中间省略标签。"""
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QLabel, QPushButton, QFileDialog, QSizePolicy, QWidget

from gui.styles import ButtonStyles, Layout


class PathPickerButton(QPushButton):
    """路径选择按钮 — 自带文件对话框的路径控件。

    按钮文本显示所选文件名（未选时显示 placeholder），tooltip 显示完整路径；
    路径由控件自身持有（`path()` / `set_path()`），调用方不再维护平行的字符串字段。

    取代「只读文本框 + 浏览按钮」两件套：音色页签的模型路径、离线页签的输入文件、
    融合/详情页签的模型文件统一使用本控件，交互一致（点一下即弹选择框）。
    """

    pathChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, title: str = "选择文件",
                 filter_str: str = "PyTorch (*.pth)", placeholder: str = "选择文件...",
                 base_dir: str | Callable[[], str] | None = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._title = title
        self._filter = filter_str
        # base_dir 允许传 callable：起始目录可能来自惰性构造的 controller
        self._base_dir = base_dir
        self._path = ""
        self.setMinimumHeight(Layout.BTN_HEIGHT_NORMAL)
        self.setStyleSheet(ButtonStyles.path_button())
        self.setToolTip(self._idle_tooltip())
        self.clicked.connect(self._pick)

    def _idle_tooltip(self) -> str:
        return f"点击{self._placeholder.rstrip('.')}"

    def _pick(self) -> None:
        base = self._base_dir() if callable(self._base_dir) else (self._base_dir or "")
        path, _ = QFileDialog.getOpenFileName(self, self._title, str(base), self._filter)
        if path:
            self.set_path(path)

    def path(self) -> str:
        """当前选中的完整路径（未选择时为空串）。"""
        return self._path

    def set_path(self, path: str, *, emit: bool = True) -> None:
        """程序化设置路径（配置回填 / 清空）；emit=False 时不触发 pathChanged。"""
        self._path = path or ""
        if self._path:
            self.setText(Path(self._path).name)
            self.setToolTip(self._path)
        else:
            self.setText(self._placeholder)
            self.setToolTip(self._idle_tooltip())
        if emit:
            self.pathChanged.emit(self._path)


class ElidedLabel(QLabel):
    """单行只读文本标签 — 超宽时按中间省略号截断，完整内容放在 tooltip。

    用于展示「由其它输入派生、用户不可编辑」的路径/结果文本：既不使用输入框
    （不可编辑，视觉上也不是输入控件），又不会因文本过长把所在页签撑宽
    （sizeHint 被 Ignored，宽度完全跟随布局分配）。
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._relayout()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        fm = self.fontMetrics()
        width = max(0, self.width() - 2)
        super().setText(fm.elidedText(self._full, Qt.TextElideMode.ElideMiddle, width))
