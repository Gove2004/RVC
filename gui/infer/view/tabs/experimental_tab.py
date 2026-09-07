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

    # ── 1. F0 中值滤波 ──
    group1 = QGroupBox("F0 中值滤波（解决破音/沙哑/带电）")
    g1 = QGridLayout(group1)
    r = 0

    win.exp_f0_median_checkbox = QCheckBox("启用")
    win.exp_f0_median_checkbox.setChecked(experimental_config.f0_median_enabled)
    win.exp_f0_median_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "f0_median_enabled", win.exp_f0_median_checkbox.isChecked())
    )
    g1.addWidget(win.exp_f0_median_checkbox, r, 0); r += 1

    win.exp_f0_median_kernel_slider = _slrow(
        win, "exp_f0_median_kernel_slider", 3.0, 9.0, 1.0,
        float(experimental_config.f0_median_kernel), fmt=".0f", unit="帧", label_w=45,
    )
    win.exp_f0_median_kernel_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "f0_median_kernel",
            int(round(_sl_value_as_float(win.exp_f0_median_kernel_slider))),
        )
    )
    g1.addWidget(QLabel("滤波窗口"), r, 0)
    g1.addWidget(win.exp_f0_median_kernel_slider, r, 1)
    g1.addWidget(win.exp_f0_median_kernel_slider_label, r, 2); r += 1

    root.addWidget(group1)

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
    g2.addWidget(win.exp_protect_threshold_slider_label, r, 2); r += 1

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
    g2.addWidget(win.exp_protect_width_slider_label, r, 2); r += 1

    root.addWidget(group2)

    # ── 3. 气息效果 ──
    group3 = QGroupBox("UV 区域合成气息噪声（解决缺少自然气息感）")
    g3 = QGridLayout(group3)
    r = 0

    win.exp_breath_checkbox = QCheckBox("启用")
    win.exp_breath_checkbox.setChecked(experimental_config.breath_enabled)
    win.exp_breath_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "breath_enabled", win.exp_breath_checkbox.isChecked())
    )
    g3.addWidget(win.exp_breath_checkbox, r, 0); r += 1

    win.exp_breath_strength_slider = _slrow(
        win, "exp_breath_strength_slider", 0.0, 1.0, 0.01,
        experimental_config.breath_strength, fmt=".2f", label_w=45,
    )
    win.exp_breath_strength_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "breath_strength",
            _sl_value_as_float(win.exp_breath_strength_slider),
        )
    )
    g3.addWidget(QLabel("气息强度"), r, 0)
    g3.addWidget(win.exp_breath_strength_slider, r, 1)
    g3.addWidget(win.exp_breath_strength_slider_label, r, 2); r += 1

    win.exp_breath_uv_threshold_slider = _slrow(
        win, "exp_breath_uv_threshold_slider", 0.0, 30.0, 1.0,
        experimental_config.breath_uv_threshold_hz, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_breath_uv_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "breath_uv_threshold_hz",
            _sl_value_as_float(win.exp_breath_uv_threshold_slider),
        )
    )
    g3.addWidget(QLabel("UV 判定阈值"), r, 0)
    g3.addWidget(win.exp_breath_uv_threshold_slider, r, 1)
    g3.addWidget(win.exp_breath_uv_threshold_slider_label, r, 2); r += 1

    root.addWidget(group3)

    # 底部说明
    hint = QLabel("提示：所有参数实时生效，无需重启。可分别开关对比效果。")
    hint.setStyleSheet("color: #888; font-size: 11px;")
    hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    root.addWidget(hint)

    root.addStretch()
    return w
