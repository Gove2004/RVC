"""推理 GUI — 模型工具 Tab（模型合并 / 模型信息 / 修正 info）。

原训练 GUI 的「工具」Tab：训练改为命令行向导后，这三项是需要图形界面（文件选择、
滑动条比例、结果回显）的功能，故迁到推理端。业务逻辑经 `ModelToolController`，
耗时操作在 `ToolRunner` 后台线程执行。
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTextEdit, QMessageBox, QFileDialog,
)

from gui.infer.controller.tool_controller import ToolRunner
from gui.styles import ButtonStyles, Layout


def build_tools_tab(win) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    layout.addWidget(_build_merge_group(win))
    layout.addWidget(_build_inspect_group(win))
    layout.addWidget(_build_fixinfo_group(win))
    layout.addStretch(1)
    return widget


# ── 共用小工具 ──

def _browse_file(parent, line_edit: QLineEdit, filter_str="PyTorch (*.pth)"):
    path, _ = QFileDialog.getOpenFileName(parent, "选择模型文件", "", filter_str)
    if path:
        line_edit.setText(path)


def _browse_button(parent, line_edit: QLineEdit, width: int | None = None) -> QPushButton:
    btn = QPushButton("浏览")
    btn.setFixedWidth(width if width is not None else Layout.BTN_WIDTH_SMALL)
    btn.setStyleSheet(ButtonStyles.small())
    btn.clicked.connect(lambda: _browse_file(parent, line_edit))
    return btn


def _run_tool(win, fn, *args, on_done):
    """统一工具执行入口：前一个未结束则等待，然后在后台线程执行。"""
    if win.tool_runner and win.tool_runner.isRunning():
        win.tool_runner.wait()
    win.tool_runner = ToolRunner(fn, *args)
    win.tool_runner.done.connect(on_done)
    win.tool_runner.start()


# ── 模型合并 ──

def _build_merge_group(win) -> QGroupBox:
    group = QGroupBox("模型合并")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    def _row(line_edit):
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(line_edit)
        row.addWidget(_browse_button(win, line_edit))
        return row

    win.merge_a = QLineEdit()
    win.merge_b = QLineEdit()
    win.merge_a.textChanged.connect(lambda: _update_merge_label(win, win.merge_slider.value()))
    win.merge_b.textChanged.connect(lambda: _update_merge_label(win, win.merge_slider.value()))
    grid.addWidget(QLabel("模型 A"), 0, 0)
    grid.addLayout(_row(win.merge_a), 0, 1, 1, 2)
    grid.addWidget(QLabel("模型 B"), 1, 0)
    grid.addLayout(_row(win.merge_b), 1, 1, 1, 2)

    win.merge_slider = QSlider(Qt.Orientation.Horizontal)
    win.merge_slider.setRange(0, 100)
    win.merge_slider.setValue(50)
    win.merge_label = QLabel("A:50% B:50%")
    win.merge_slider.valueChanged.connect(lambda v: _update_merge_label(win, v))
    grid.addWidget(QLabel("比例"), 2, 0)
    grid.addWidget(win.merge_slider, 2, 1)
    grid.addWidget(win.merge_label, 2, 2)

    win.merge_name = QLineEdit("merged")
    win.btn_merge = QPushButton("合并")
    win.btn_merge.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    win.btn_merge.setStyleSheet(ButtonStyles.small())
    win.btn_merge.clicked.connect(lambda: _run_merge(win))
    name_row = QHBoxLayout()
    name_row.setSpacing(4)
    name_row.addWidget(win.merge_name)
    name_row.addWidget(win.btn_merge)
    grid.addWidget(QLabel("输出名"), 3, 0)
    grid.addLayout(name_row, 3, 1, 1, 2)

    return group


def _update_merge_label(win, value: int):
    a = Path(win.merge_a.text()).stem or "A"
    b = Path(win.merge_b.text()).stem or "B"
    win.merge_label.setText(f"{a}:{value}% {b}:{100 - value}%")


def _run_merge(win):
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
    out = win.model_tools.merge_output_path(name)
    ratio = win.merge_slider.value() / 100.0
    win.btn_merge.setEnabled(False)
    win.btn_merge.setStyleSheet(ButtonStyles.secondary())
    _run_tool(win, win.model_tools.merge_models, a, b, ratio, out,
              on_done=lambda ok, msg: _on_merge_done(win, ok, msg))


def _on_merge_done(win, success, message):
    win.btn_merge.setEnabled(True)
    win.btn_merge.setStyleSheet(ButtonStyles.small())
    if success:
        name = win.merge_name.text().strip()
        QMessageBox.information(win, "完成", f"模型已保存到 assets/models/{name}.pth")
    else:
        QMessageBox.critical(win, "合并失败", message)


# ── 模型信息（查看 + 改 zip 原名）──

def _build_inspect_group(win) -> QGroupBox:
    group = QGroupBox("模型信息")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    win.inspect_path = QLineEdit()
    grid.addWidget(QLabel("模型文件"), 0, 0)
    grid.addWidget(win.inspect_path, 0, 1)
    grid.addWidget(_browse_button(win, win.inspect_path), 0, 2)

    btn_inspect = QPushButton("查看")
    btn_inspect.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_inspect.setStyleSheet(ButtonStyles.small())
    btn_inspect.clicked.connect(lambda: _run_inspect(win))
    grid.addWidget(btn_inspect, 0, 3)

    win.inspect_result = QTextEdit()
    win.inspect_result.setReadOnly(True)
    win.inspect_result.setFixedHeight(90)
    grid.addWidget(win.inspect_result, 1, 0, 1, 4)

    win.archive_name_edit = QLineEdit()
    win.archive_name_edit.setPlaceholderText("点击查看后自动填入原名，可修改为新名")
    btn_rename = QPushButton("修改 zip 原名")
    btn_rename.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_rename.setStyleSheet(ButtonStyles.small())
    btn_rename.clicked.connect(lambda: _run_rename_archive(win))
    grid.addWidget(QLabel("zip 原名"), 2, 0)
    grid.addWidget(win.archive_name_edit, 2, 1)
    grid.addWidget(btn_rename, 2, 2, 1, 2)

    return group


def _run_inspect(win):
    path = win.inspect_path.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请选择模型文件")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    win.inspect_result.setText("加载中...")
    _run_tool(win, win.model_tools.inspect_model, path,
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
    path = win.inspect_path.text().strip()
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
    _run_tool(win, win.model_tools.change_archive_name, path, new_name,
              on_done=lambda ok, msg: _on_rename_done(win, ok, msg, path))


def _on_rename_done(win, success, message, path):
    if success:
        QMessageBox.information(win, "完成", f"zip 原名已修改为：{message}")
        if Path(win.inspect_path.text().strip()).resolve() == Path(path).resolve():
            _run_inspect(win)
    else:
        QMessageBox.critical(win, "失败", message)


# ── 修正模型信息（info）──

def _build_fixinfo_group(win) -> QGroupBox:
    group = QGroupBox("修正模型信息（info）")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    win.fix_path = QLineEdit()
    grid.addWidget(QLabel("模型文件"), 0, 0)
    grid.addWidget(win.fix_path, 0, 1)
    grid.addWidget(_browse_button(win, win.fix_path), 0, 2)

    win.model_info = QLineEdit()
    win.model_info.setPlaceholderText("输入新的模型信息，例如 exp01（自动去掉误带的 .pth）")
    btn_apply = QPushButton("应用")
    btn_apply.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_apply.setStyleSheet(ButtonStyles.small())
    btn_apply.clicked.connect(lambda: _apply_model_info(win))
    grid.addWidget(QLabel("新模型信息"), 1, 0)
    grid.addWidget(win.model_info, 1, 1)
    grid.addWidget(btn_apply, 1, 2)

    return group


def _apply_model_info(win):
    path = win.fix_path.text().strip()
    model_info = win.model_info.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请选择模型文件")
        return
    if not model_info:
        QMessageBox.warning(win, "提示", "请输入新的模型信息")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    _run_tool(win, win.model_tools.change_model_info, path, model_info,
              on_done=lambda ok, msg: _on_fixinfo_done(win, ok, msg, path))


def _on_fixinfo_done(win, success, message, path):
    if success:
        QMessageBox.information(win, "完成", "模型信息已更新")
        # 若在检视框里也选了同一文件，刷新显示
        if Path(win.inspect_path.text().strip()).resolve() == Path(path).resolve():
            _run_inspect(win)
    else:
        QMessageBox.critical(win, "失败", message)
