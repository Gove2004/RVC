"""训练 GUI 通用组件"""

from PySide6.QtWidgets import QFileDialog, QLineEdit


def browse_directory(parent, line_edit: QLineEdit):
    path = QFileDialog.getExistingDirectory(parent, "选择音频目录")
    if path:
        line_edit.setText(path)


def browse_file(parent, line_edit: QLineEdit, filter_str="PyTorch (*.pth)"):
    path, _ = QFileDialog.getOpenFileName(parent, "选择模型文件", "", filter_str)
    if path:
        line_edit.setText(path)
