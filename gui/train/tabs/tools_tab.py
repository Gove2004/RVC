"""工具 Tab — 模型合并 + 模型信息"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTextEdit, QMessageBox,
)

from gui.train.widgets import ToolThread, browse_file


def build_tools_tab(win) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)
    layout.addWidget(_build_merge_group(win))
    layout.addWidget(_build_inspect_group(win))
    layout.addStretch(1)
    return widget


def _build_merge_group(win) -> QGroupBox:
    group = QGroupBox("模型合并")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    def _row(parent, line_edit):
        row = QHBoxLayout()
        row.setSpacing(4)
        btn = QPushButton("浏览")
        btn.setFixedWidth(33)  # 50/1.5≈33
        btn.clicked.connect(lambda: browse_file(parent, line_edit))
        row.addWidget(line_edit)
        row.addWidget(btn)
        return row

    win.merge_a = QLineEdit()
    win.merge_b = QLineEdit()
    win.merge_a.textChanged.connect(lambda: _on_merge_slider(win, win.merge_slider.value()))
    win.merge_b.textChanged.connect(lambda: _on_merge_slider(win, win.merge_slider.value()))
    grid.addWidget(QLabel("模型 A"), 0, 0)
    grid.addLayout(_row(win, win.merge_a), 0, 1, 1, 2)
    grid.addWidget(QLabel("模型 B"), 1, 0)
    grid.addLayout(_row(win, win.merge_b), 1, 1, 1, 2)

    win.merge_slider = QSlider(Qt.Orientation.Horizontal)
    win.merge_slider.setRange(0, 100)
    win.merge_slider.setValue(50)
    win.merge_label = QLabel("A:50% B:50%")
    win.merge_slider.valueChanged.connect(lambda v: _on_merge_slider(win, v))
    grid.addWidget(QLabel("比例"), 2, 0)
    grid.addWidget(win.merge_slider, 2, 1)
    grid.addWidget(win.merge_label, 2, 2)

    win.merge_name = QLineEdit("merged")
    win.btn_merge = QPushButton("合并")
    win.btn_merge.setFixedWidth(33)  # 50/1.5≈33
    win.btn_merge.clicked.connect(lambda: _run_merge(win))
    name_row = QHBoxLayout()
    name_row.setSpacing(4)
    name_row.addWidget(win.merge_name)
    name_row.addWidget(win.btn_merge)
    grid.addWidget(QLabel("输出名"), 3, 0)
    grid.addLayout(name_row, 3, 1, 1, 2)

    return group


def _build_inspect_group(win) -> QGroupBox:
    group = QGroupBox("模型信息")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    win.inspect_path = QLineEdit()
    btn_browse = QPushButton("浏览")
    btn_browse.setFixedWidth(33)  # 50/1.5≈33
    btn_browse.clicked.connect(lambda: browse_file(win, win.inspect_path))
    grid.addWidget(QLabel("模型文件"), 0, 0)
    grid.addWidget(win.inspect_path, 0, 1)
    grid.addWidget(btn_browse, 0, 2)

    btn_inspect = QPushButton("查看")
    btn_inspect.setFixedWidth(33)  # 50/1.5≈33
    btn_inspect.clicked.connect(lambda: _run_inspect(win))
    grid.addWidget(btn_inspect, 0, 3)

    win.inspect_result = QTextEdit()
    win.inspect_result.setReadOnly(True)
    win.inspect_result.setFixedHeight(90)
    grid.addWidget(win.inspect_result, 1, 0, 1, 4)

    return group


def _on_merge_slider(win, value):
    a = Path(win.merge_a.text()).stem or "A"
    b = Path(win.merge_b.text()).stem or "B"
    win.merge_label.setText(f"{a}:{value}% {b}:{100 - value}%")


def _run_merge(win):
    from rvc.train.ckpt_utils import merge_models

    a = win.merge_a.text().strip()
    b = win.merge_b.text().strip()
    name = win.merge_name.text().strip()
    if not a or not b or not name:
        QMessageBox.warning(win, "提示", "请填写所有字段")
        return
    if not Path(a).exists():
        QMessageBox.warning(win, "提示", "模型 A 不存在")
        return
    if not Path(b).exists():
        QMessageBox.warning(win, "提示", "模型 B 不存在")
        return
    out = str(Path("assets/weights") / f"{name}.pth")
    ratio = win.merge_slider.value() / 100.0
    win.btn_merge.setEnabled(False)
    win.btn_merge.setStyleSheet("QPushButton{background:#3b82f6;color:white;border:none;padding:3px;border-radius:3px;font-size:11px}")
    if win._tool_thread and win._tool_thread.isRunning():
        win._tool_thread.wait()
    win._tool_thread = ToolThread(merge_models, a, b, ratio, out)
    win._tool_thread.done.connect(lambda ok, msg: _on_merge_done(win, ok, msg))
    win._tool_thread.start()


def _on_merge_done(win, success, message):
    win.btn_merge.setEnabled(True)
    win.btn_merge.setStyleSheet("")
    if success:
        name = win.merge_name.text().strip()
        QMessageBox.information(win, "完成", f"模型已保存到 assets/weights/{name}.pth")
    else:
        QMessageBox.critical(win, "合并失败", message)


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
    else:
        win.inspect_result.setText("")
        QMessageBox.critical(win, "错误", result)
