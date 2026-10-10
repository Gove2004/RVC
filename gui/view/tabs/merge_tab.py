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
from gui.view.widgets import PathPickerButton


def build_merge_tab(win) -> QWidget:
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    # 与「音色 / 参数」页签同一套列比例（滑块独占第 1 列、值标签 80px 占第 2 列），
    # 三个页签的滑块因此逐像素对齐：实测页宽 525 时融合 x=112/宽 319，
    # 参数 x=112/宽 319，音色 x=110/宽 312（音色第 1 列含 RangeSlider 最小宽 180，
    # 分配略有 7px 残差，视觉不可见）。
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
    # 版式与「音色 / 参数」页签的滑块行完全一致：标签 | 滑块(col1) | 值(col2)。
    # 值列必须是**定宽标签**（同 _create_slider_row(label_w=80)）：定宽后列宽
    # 才与其它页签相同，滑块也就拿到同样的宽度与左边界；换成长度随内容变化的
    # 标签，col2 会被挤掉、多余宽度全归滑块（这正是之前「宽度不太对」的成因）。
    win.merge_slider = QSlider(Qt.Orientation.Horizontal)
    win.merge_slider.setRange(0, 100)
    win.merge_slider.setValue(50)
    win.merge_slider.setFixedHeight(18)
    win.merge_slider.valueChanged.connect(lambda v: _update_merge_label(win, v))

    win.merge_label = QLabel()
    win.merge_label.setFixedWidth(80)
    win.merge_label.setAlignment(Qt.AlignCenter)

    g.addWidget(QLabel("融合比例"), r, 0)
    g.addWidget(win.merge_slider, r, 1)
    g.addWidget(win.merge_label, r, 2)
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
    """刷新比例值标签 — 只显示百分比（宽 80px），完整模型名放 tooltip。

    模型名在上面两个路径按钮上已经可见，值标签不重复占用宽度。
    """
    a = Path(win.merge_a.path()).stem or "A"
    b = Path(win.merge_b.path()).stem or "B"
    win.merge_label.setText(f"{value}% / {100 - value}%")
    win.merge_label.setToolTip(f"{a}: {value}%    {b}: {100 - value}%")


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
