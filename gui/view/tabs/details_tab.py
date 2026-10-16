"""详情 Tab — 查看模型信息 / 修改 zip 原名。

原「工具」Tab 的模型信息部分；训练 GUI 退场后迁入推理端。业务逻辑经
`ModelToolController`，耗时操作在 `ToolRunner` 后台线程执行。
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
)

from gui.controller.tool_controller import run_tool_async
from gui.styles import ButtonStyles, Layout
from gui.view.widgets import PathPickerButton


def build_details_tab(win) -> QWidget:
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(1, 1)
    r = 0

    # ── 模型文件（自带文件对话框的路径按钮，与音色页签一致）+ 查看 ──
    win.inspect_path = PathPickerButton(
        win, title="选择模型文件", placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )
    btn_inspect = QPushButton("查看")
    btn_inspect.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_inspect.setStyleSheet(ButtonStyles.small())
    btn_inspect.clicked.connect(lambda: _run_inspect(win))

    g.addWidget(QLabel("模型文件"), r, 0)
    g.addWidget(win.inspect_path, r, 1)
    g.addWidget(btn_inspect, r, 2)
    r += 1

    # ── 模型信息展示 ──
    win.inspect_result = QTextEdit()
    win.inspect_result.setReadOnly(True)
    win.inspect_result.setMinimumHeight(120)
    g.addWidget(QLabel("模型信息"), r, 0, Qt.AlignmentFlag.AlignTop)
    g.addWidget(win.inspect_result, r, 1, 1, 2)
    r += 1

    # ── 修改 zip 原名 ──
    win.archive_name_edit = QLineEdit()
    win.archive_name_edit.setPlaceholderText("点击查看后自动填入原名，可修改为新名")
    btn_rename = QPushButton("修改原名")
    btn_rename.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_rename.setStyleSheet(ButtonStyles.small())
    btn_rename.clicked.connect(lambda: _run_rename_archive(win))

    g.addWidget(QLabel("原名"), r, 0)
    g.addWidget(win.archive_name_edit, r, 1)
    g.addWidget(btn_rename, r, 2)
    r += 1

    g.setRowStretch(r, 1)
    return w


def _run_inspect(win):
    path = win.inspect_path.path().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请选择模型文件")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    win.inspect_result.setText("加载中...")
    run_tool_async(win, win.model_tools.inspect_model, path,
              on_done=lambda ok, msg: _on_inspect_done(win, ok, msg))


def _on_inspect_done(win, success, result):
    if success:
        win.inspect_result.setText(result)
        # 把当前 zip 原名填进改名框，方便直接改（解析「zip 原名:」行，
        # 首行优先展示 info，故不能读首行）
        for line in result.splitlines():
            if "zip 原名:" in line:
                win.archive_name_edit.setText(line.split("zip 原名:", 1)[1].strip())
                break
    else:
        win.inspect_result.setText("")
        QMessageBox.critical(win, "错误", result)


def _run_rename_archive(win):
    path = win.inspect_path.path().strip()
    new_name = win.archive_name_edit.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请先选择模型文件")
        return
    if not new_name:
        QMessageBox.warning(win, "提示", "请输入新的 zip 原名")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    run_tool_async(win, win.model_tools.change_archive_name, path, new_name,
              on_done=lambda ok, msg: _on_rename_done(win, ok, msg, path))


def _on_rename_done(win, success, message, path):
    if success:
        QMessageBox.information(win, "完成", f"zip 原名已修改为：{message}")
        if Path(win.inspect_path.path().strip()).resolve() == Path(path).resolve():
            _run_inspect(win)
    else:
        QMessageBox.critical(win, "失败", message)
