"""详情 Tab — 模型信息查看。

选择模型文件后自动解析并显示信息（无需点击按钮）。
业务逻辑经 `ModelToolController`，耗时操作在 `ToolRunner` 后台线程执行。
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QTextEdit, QMessageBox,
)

from gui.controller.tool_controller import run_tool_async
from gui.view.widgets import PathPickerButton


def build_details_tab(win) -> QWidget:
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(1, 1)
    r = 0

    # ── 模型文件（自带文件对话框的路径按钮，与音色页签一致）──
    # 选定路径即自动解析（pathChanged 驱动），不需要额外的「查看」按钮。
    win.inspect_path = PathPickerButton(
        win, title="选择模型文件", placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )
    win.inspect_path.pathChanged.connect(
        lambda path: _run_inspect(win) if path.strip() else None
    )

    g.addWidget(QLabel("模型文件"), r, 0)
    g.addWidget(win.inspect_path, r, 1, 1, 2)
    r += 1

    # ── 模型信息展示 ──
    win.inspect_result = QTextEdit()
    win.inspect_result.setReadOnly(True)
    win.inspect_result.setMinimumHeight(120)
    g.addWidget(QLabel("模型信息"), r, 0, Qt.AlignmentFlag.AlignTop)
    g.addWidget(win.inspect_result, r, 1, 1, 2)
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
    else:
        win.inspect_result.setText("")
        QMessageBox.critical(win, "错误", result)
