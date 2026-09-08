"""实验功能 Tab — 辅音保护 / 破音保护 / 辅音软阈值"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout,
)
from PySide6.QtCore import Qt

from gui.infer.view.widgets import _slrow, _sl_value_as_float
from rvc.core.experimental import experimental_config


def _group_with_toggle(title: str, checkbox: QCheckBox) -> tuple[QGroupBox, QGridLayout]:
    """创建带标题+启用勾选框的 GroupBox，勾选框在标题文字右侧。"""
    box = QGroupBox()
    outer = QVBoxLayout(box)
    outer.setContentsMargins(8, 4, 8, 8)

    # 标题行：文字 + 勾选框
    header = QHBoxLayout()
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-weight: bold;")
    header.addWidget(title_lbl)
    header.addWidget(checkbox)
    header.addStretch()
    outer.addLayout(header)

    # 内容网格
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    outer.addLayout(grid)
    return box, grid


def build_experimental_tab(win):
    w = QWidget()
    root = QVBoxLayout(w)

    # ── 1. 辅音保护强度 ──
    group1 = QGroupBox("辅音保护强度")
    g1 = QGridLayout(group1)
    r = 0
    win.protect_slider = _slrow(win, "protect_slider", 0.0, 1.0, 0.01, 0.5)
    g1.addWidget(QLabel("保护强度"), r, 0)
    g1.addWidget(win.protect_slider, r, 1)
    g1.addWidget(win.protect_label, r, 2); r += 1
    root.addWidget(group1)

    # ── 2. 辅音保护软阈值 ──
    win.exp_protect_soft_checkbox = QCheckBox("启用")
    win.exp_protect_soft_checkbox.setChecked(experimental_config.protect_soft_enabled)
    win.exp_protect_soft_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "protect_soft_enabled", win.exp_protect_soft_checkbox.isChecked())
    )
    group2, g2 = _group_with_toggle("辅音保护软阈值", win.exp_protect_soft_checkbox)
    r = 0

    win.exp_protect_threshold_slider = _slrow(
        win, "exp_protect_threshold_slider", 0.0, 30.0, 1.0,
        experimental_config.protect_soft_threshold_hz, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_threshold_hz",
            _sl_value_as_float(win.exp_protect_threshold_slider),
        )
    )
    g2.addWidget(QLabel("过渡中心"), r, 0)
    g2.addWidget(win.exp_protect_threshold_slider, r, 1)
    g2.addWidget(win.exp_protect_threshold_label, r, 2); r += 1

    win.exp_protect_width_slider = _slrow(
        win, "exp_protect_width_slider", 5.0, 50.0, 1.0,
        experimental_config.protect_soft_width, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_width_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_width",
            _sl_value_as_float(win.exp_protect_width_slider),
        )
    )
    g2.addWidget(QLabel("过渡宽度"), r, 0)
    g2.addWidget(win.exp_protect_width_slider, r, 1)
    g2.addWidget(win.exp_protect_width_label, r, 2); r += 1

    root.addWidget(group2)

    # ── 3. 破音保护 ──
    win.break_enable_checkbox = QCheckBox("启用")
    group3, g3 = _group_with_toggle("破音保护", win.break_enable_checkbox)
    r = 0
    win.break_src_hz_slider = _slrow(win, "break_src_hz_slider", 200.0, 400.0, 5.0, 300.0, fmt=".0f", unit="Hz", label_w=45)
    g3.addWidget(QLabel("破音临界"), r, 0)
    g3.addWidget(win.break_src_hz_slider, r, 1)
    g3.addWidget(win.break_src_hz_label, r, 2); r += 1
    root.addWidget(group3)

    root.addStretch()
    return w

