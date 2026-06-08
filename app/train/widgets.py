"""训练 GUI 通用组件"""
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QLineEdit


class ToolThread(QThread):
    """后台执行模型合并等耗时操作"""
    done = Signal(bool, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.done.emit(True, result if isinstance(result, str) else "操作完成")
        except Exception as e:
            self.done.emit(False, str(e))


def browse_directory(parent, line_edit: QLineEdit):
    path = QFileDialog.getExistingDirectory(parent, "选择音频目录")
    if path:
        line_edit.setText(path)


def browse_file(parent, line_edit: QLineEdit, filter_str="PyTorch (*.pth)"):
    path, _ = QFileDialog.getOpenFileName(parent, "选择模型文件", "", filter_str)
    if path:
        line_edit.setText(path)
