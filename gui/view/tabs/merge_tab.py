"""融合 Tab — 模型合并：两个模型按比例融合为一个新模型。

原「工具」Tab 的模型合并部分；训练 GUI 退场后随其余模型工具一起迁入推理端。
业务逻辑经 `ModelToolController`，耗时操作在 `ToolRunner` 后台线程执行。
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QSlider,
)

from gui.controller.tool_controller import ToolRunner
from gui.styles import ButtonStyles
from gui.styles.colors import Colors
from gui.view.widgets import ElidedLabel, PathPickerButton


def build_merge_tab(win) -> QWidget:
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    # 与「音色 / 参数」页签同一套列比例：滑块独占第 1 列，左边界与宽度
    # 与其它页签的滑块一致（实测 560px 宽下 x=120 / 宽 346，音色页签 x=118 / 宽 338）
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 模型 A / 模型 B（自带文件对话框的路径按钮，与音色页签一致）──
    win.merge_a = PathPickerButton(
        win, title="选择模型 A", placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )
    win.merge_a.pathChanged.connect(lambda _: _update_merge_label(win, win.merge_slider.value()))
    g.addWidget(QLabel("模型 A"), r, 0)
    g.addWidget(win.merge_a, r, 1, 1, 2)
    r += 1

    win.merge_b = PathPickerButton(
        win, title="选择模型 B", placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )
    win.merge_b.pathChanged.connect(lambda _: _update_merge_label(win, win.merge_slider.value()))
    g.addWidget(QLabel("模型 B"), r, 0)
    g.addWidget(win.merge_b, r, 1, 1, 2)
    r += 1

    # ── 融合比例（A 的占比；B = 100 - A）──
    win.merge_slider = QSlider(Qt.Orientation.Horizontal)
    win.merge_slider.setRange(0, 100)
    win.merge_slider.setValue(50)
    win.merge_slider.setFixedHeight(18)
    win.merge_slider.valueChanged.connect(lambda v: _update_merge_label(win, v))
    g.addWidget(QLabel("融合比例"), r, 0)
    g.addWidget(win.merge_slider, r, 1)
    r += 1

    # 比例文字单独一行（在滑块下方）：可容纳长模型名而不挤压滑块宽度；
    # 超宽时中间省略，完整内容在 tooltip 里。
    win.merge_label = ElidedLabel()
    win.merge_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
    g.addWidget(win.merge_label, r, 1, 1, 2)
    r += 1

    # ── 模型名（输出文件名，落 assets/models/<名>.pth）──
    win.merge_name = QLineEdit("merged")
    win.merge_name.setPlaceholderText("输出模型名（不含 .pth）")
    g.addWidget(QLabel("模型名"), r, 0)
    g.addWidget(win.merge_name, r, 1, 1, 2)
    r += 1

    # ── 开始融合 ──
    win.btn_merge = QPushButton("开始融合")
    win.btn_merge.setMinimumHeight(32)
    win.btn_merge.setStyleSheet(ButtonStyles.primary())
    win.btn_merge.clicked.connect(lambda: _run_merge(win))
    g.addWidget(win.btn_merge, r, 0, 1, 3)
    r += 1

    # ── 状态文本 ──
    win.merge_status = QLabel("")
    g.addWidget(win.merge_status, r, 0, 1, 3)
    r += 1

    _update_merge_label(win, win.merge_slider.value())

    g.setRowStretch(r, 1)
    return w


def _update_merge_label(win, value: int):
    a = Path(win.merge_a.path()).stem or "A"
    b = Path(win.merge_b.path()).stem or "B"
    win.merge_label.setText(f"{a}:{value}% {b}:{100 - value}%")


def _set_merge_status(win, text: str, color: str = "") -> None:
    win.merge_status.setStyleSheet(f"color: {color};" if color else "")
    win.merge_status.setText(text)


def _run_merge(win):
    a = win.merge_a.path().strip()
    b = win.merge_b.path().strip()
    name = win.merge_name.text().strip()
    if not a or not b or not name:
        _set_merge_status(win, "请填写所有字段", Colors.ERROR)
        return
    if not Path(a).exists():
        _set_merge_status(win, "模型 A 不存在", Colors.ERROR)
        return
    if not Path(b).exists():
        _set_merge_status(win, "模型 B 不存在", Colors.ERROR)
        return
    out = win.model_tools.merge_output_path(name)
    ratio = win.merge_slider.value() / 100.0
    win.btn_merge.setEnabled(False)
    win.btn_merge.setStyleSheet(ButtonStyles.secondary())
    _set_merge_status(win, "融合中...", Colors.TEXT_SECONDARY)
    _run_tool(win, win.model_tools.merge_models, a, b, ratio, out,
              on_done=lambda ok, msg: _on_merge_done(win, ok, msg, name))


def _on_merge_done(win, success, message, name):
    win.btn_merge.setEnabled(True)
    win.btn_merge.setStyleSheet(ButtonStyles.primary())
    if success:
        _set_merge_status(win, f"完成：已保存到 assets/models/{name}.pth", Colors.SUCCESS)
    else:
        _set_merge_status(win, f"失败：{message}", Colors.ERROR)


def _run_tool(win, fn, *args, on_done):
    """统一工具执行入口：前一个未结束则等待，然后在后台线程执行。"""
    if win.tool_runner and win.tool_runner.isRunning():
        win.tool_runner.wait()
    win.tool_runner = ToolRunner(fn, *args)
    win.tool_runner.done.connect(on_done)
    win.tool_runner.start()
