"""实验功能 Tab — 辅音保护 / 音域映射"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QGroupBox, QVBoxLayout,
)

from gui.infer.view.widgets import _slrow, _sl_value_as_float, RangeSlider
from rvc.core.experimental import experimental_config


def build_experimental_tab(win):
    w = QWidget()
    root = QVBoxLayout(w)

    # ── 1. 辅音保护（渐变式，始终生效） ──
    group1 = QGroupBox("辅音保护")
    g1 = QGridLayout(group1)
    g1.setHorizontalSpacing(8)
    r = 0

    # RMVPE 阈值
    win.exp_rmvpe_threshold_slider = _slrow(
        win, "exp_rmvpe_threshold_slider", 0.01, 0.10, 0.01,
        experimental_config.rmvpe_threshold, fmt=".2f", label_w=45,
    )
    win.exp_rmvpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "rmvpe_threshold",
            _sl_value_as_float(win.exp_rmvpe_threshold_slider),
        )
    )
    g1.addWidget(QLabel("RMVPE 阈值"), r, 0)
    g1.addWidget(win.exp_rmvpe_threshold_slider, r, 1)
    g1.addWidget(win.exp_rmvpe_threshold_label, r, 2); r += 1

    # FCPE 阈值
    win.exp_fcpe_threshold_slider = _slrow(
        win, "exp_fcpe_threshold_slider", 0.01, 0.10, 0.01,
        experimental_config.fcpe_confidence_threshold, fmt=".2f", label_w=45,
    )
    win.exp_fcpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "fcpe_confidence_threshold",
            _sl_value_as_float(win.exp_fcpe_threshold_slider),
        )
    )
    g1.addWidget(QLabel("FCPE 阈值"), r, 0)
    g1.addWidget(win.exp_fcpe_threshold_slider, r, 1)
    g1.addWidget(win.exp_fcpe_threshold_label, r, 2); r += 1

    # 过渡中心
    win.exp_protect_threshold_slider = _slrow(
        win, "exp_protect_threshold_slider", 0.0, 60.0, 1.0,
        experimental_config.protect_soft_threshold_hz, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_threshold_hz",
            _sl_value_as_float(win.exp_protect_threshold_slider),
        )
    )
    g1.addWidget(QLabel("过渡中心"), r, 0)
    g1.addWidget(win.exp_protect_threshold_slider, r, 1)
    g1.addWidget(win.exp_protect_threshold_label, r, 2); r += 1

    # 过渡宽度
    win.exp_protect_width_slider = _slrow(
        win, "exp_protect_width_slider", 5.0, 80.0, 1.0,
        experimental_config.protect_soft_width, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_width_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_width",
            _sl_value_as_float(win.exp_protect_width_slider),
        )
    )
    g1.addWidget(QLabel("过渡宽度"), r, 0)
    g1.addWidget(win.exp_protect_width_slider, r, 1)
    g1.addWidget(win.exp_protect_width_label, r, 2); r += 1

    # 保护强度
    win.protect_slider = _slrow(win, "protect_slider", 0.0, 1.0, 0.01, 0.5)
    g1.addWidget(QLabel("保护强度"), r, 0)
    g1.addWidget(win.protect_slider, r, 1)
    g1.addWidget(win.protect_label, r, 2); r += 1

    root.addWidget(group1)

    # ── 2. 音域映射（半音尺度，始终生效，替代固定 pitch 偏移） ──
    group_pm = QGroupBox("音域映射（半音尺度，保持音程）")
    gpm = QGridLayout(group_pm)
    gpm.setHorizontalSpacing(8)
    r = 0

    # 原声音域（双滑块范围控件）
    win.exp_pitch_map_src_range = RangeSlider(
        20.0, 1000.0, 5.0,
        experimental_config.pitch_map_src_min, experimental_config.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    win.exp_pitch_map_src_range.rangeChanged.connect(
        lambda low, high: (
            setattr(experimental_config, "pitch_map_src_min", low),
            setattr(experimental_config, "pitch_map_src_max", high),
        )
    )
    gpm.addWidget(QLabel("原声音域"), r, 0)
    gpm.addWidget(win.exp_pitch_map_src_range, r, 1, 1, 2); r += 1

    # 目标音域（双滑块范围控件）
    win.exp_pitch_map_dst_range = RangeSlider(
        20.0, 1000.0, 5.0,
        experimental_config.pitch_map_dst_min, experimental_config.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    win.exp_pitch_map_dst_range.rangeChanged.connect(
        lambda low, high: (
            setattr(experimental_config, "pitch_map_dst_min", low),
            setattr(experimental_config, "pitch_map_dst_max", high),
        )
    )
    gpm.addWidget(QLabel("目标音域"), r, 0)
    gpm.addWidget(win.exp_pitch_map_dst_range, r, 1, 1, 2); r += 1

    root.addWidget(group_pm)
    root.addStretch()
    return w
