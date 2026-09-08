"""实验功能 Tab — 音质优化参数实时调节（F0中值滤波 / 辅音软阈值 / 气息噪声）"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QGroupBox, QVBoxLayout,
)
from PySide6.QtCore import Qt

from gui.infer.view.widgets import _slrow, _sl_value_as_float
from rvc.core.experimental import experimental_config


def build_experimental_tab(win):
    w = QWidget()
    root = QVBoxLayout(w)

    # ── 2. 辅音保护软阈值 ──
    group2 = QGroupBox("辅音保护软阈值（解决咬字不清）")
    g2 = QGridLayout(group2)
    r = 0

    win.exp_protect_soft_checkbox = QCheckBox("启用")
    win.exp_protect_soft_checkbox.setChecked(experimental_config.protect_soft_enabled)
    win.exp_protect_soft_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "protect_soft_enabled", win.exp_protect_soft_checkbox.isChecked())
    )
    g2.addWidget(win.exp_protect_soft_checkbox, r, 0); r += 1

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

    root.addStretch()
    return w

