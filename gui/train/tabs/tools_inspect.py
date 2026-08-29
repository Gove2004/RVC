"""训练工具 Tab — 模型信息查看 + 修改 zip 原名（从 tools_tab.py 拆出）"""
from pathlib import Path

from PySide6.QtWidgets import (
    QGroupBox, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
)

from gui.train.widgets import ToolThread, browse_file
from gui.styles import ButtonStyles, Layout


def build_group(win) -> QGroupBox:
    group = QGroupBox("模型信息")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    win.inspect_path = QLineEdit()
    btn_browse = QPushButton("浏览")
    btn_browse.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_browse.setStyleSheet(ButtonStyles.small())
    btn_browse.clicked.connect(lambda: browse_file(win, win.inspect_path))
    grid.addWidget(QLabel("模型文件"), 0, 0)
    grid.addWidget(win.inspect_path, 0, 1)
    grid.addWidget(btn_browse, 0, 2)

    btn_inspect = QPushButton("查看")
    btn_inspect.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_inspect.setStyleSheet(ButtonStyles.small())
    btn_inspect.clicked.connect(lambda: _run_inspect(win))
    grid.addWidget(btn_inspect, 0, 3)

    win.inspect_result = QTextEdit()
    win.inspect_result.setReadOnly(True)
    win.inspect_result.setFixedHeight(90)
    grid.addWidget(win.inspect_result, 1, 0, 1, 4)

    win.rename_edit = QLineEdit()
    win.rename_edit.setPlaceholderText("点击查看后自动填入原名，可修改为新名")
    btn_rename = QPushButton("修改 zip 原名")
    btn_rename.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_rename.setStyleSheet(ButtonStyles.small())
    btn_rename.clicked.connect(lambda: _run_rename_zip(win))
    grid.addWidget(QLabel("zip 原名"), 2, 0)
    grid.addWidget(win.rename_edit, 2, 1)
    grid.addWidget(btn_rename, 2, 2, 1, 2)

    return group


def _run_inspect(win):
    from rvc.train.ckpt_utils import inspect_model

    path = win.inspect_path.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请选择模型文件")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    win.inspect_result.setText("加载中...")
    if win._tool_thread and win._tool_thread.isRunning():
        win._tool_thread.wait()
    win._tool_thread = ToolThread(inspect_model, path)
    win._tool_thread.done.connect(lambda ok, msg: _on_inspect_done(win, ok, msg))
    win._tool_thread.start()


def _on_inspect_done(win, success, result):
    if success:
        win.inspect_result.setText(result)
        # 把当前 zip 原名填进改名框，方便直接改
        first = result.splitlines()[0]
        if "真名/模型信息:" in first:
            win.rename_edit.setText(first.split("真名/模型信息:", 1)[1].strip())
    else:
        win.inspect_result.setText("")
        QMessageBox.critical(win, "错误", result)


def _run_rename_zip(win):
    from rvc.train.ckpt_utils import change_archive_name

    path = win.inspect_path.text().strip()
    new_name = win.rename_edit.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请先选择模型文件")
        return
    if not new_name:
        QMessageBox.warning(win, "提示", "请输入新的 zip 原名")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    if win._tool_thread and win._tool_thread.isRunning():
        win._tool_thread.wait()
    win._tool_thread = ToolThread(change_archive_name, path, new_name)
    win._tool_thread.done.connect(lambda ok, msg: _on_rename_done(win, ok, msg, path))
    win._tool_thread.start()


def _on_rename_done(win, success, message, path):
    if success:
        QMessageBox.information(win, "完成", f"zip 原名已修改为：{message}")
        if Path(win.inspect_path.text().strip()).resolve() == Path(path).resolve():
            _run_inspect(win)
    else:
        QMessageBox.critical(win, "失败", message)
