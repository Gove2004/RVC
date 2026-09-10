"""实验功能 Tab — 辅音保护 / 音域映射（无分组，直接罗列）"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel

from gui.infer.view.widgets import _slrow, _sl_value_as_float, RangeSlider
from rvc.core.experimental import experimental_config


def _range_row(win, attr, min_val, max_val, step, low_val, high_val,
               fmt=".0f", unit="Hz", label_w=80):
    """创建「双滑块范围 + 自动格式化值标签」并挂到 win.<attr> / win.<attr>_label。

    与 _slrow 类似，但用于 RangeSlider 双滑块。标签显示 "下限-上限 单位"。
    """
    rs = RangeSlider(min_val, max_val, step, low_val, high_val, fmt=fmt, unit=unit)
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignCenter)

    def _fmt(low, high):
        return f"{low:{fmt}}-{high:{fmt}}{unit}"

    lbl.setText(_fmt(rs.low(), rs.high()))
    rs.rangeChanged.connect(lambda low, high: lbl.setText(_fmt(low, high)))
    setattr(win, attr, rs)
    label_attr = attr[:-6] + "_label" if attr.endswith("_range") else attr + "_label"
    setattr(win, label_attr, lbl)
    return rs


def build_experimental_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    # 列宽比例：标签 20%、滑动条 65%、值 15%
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 1. RMVPE 阈值 ──
    win.exp_rmvpe_threshold_slider = _slrow(
        win, "exp_rmvpe_threshold_slider", 0.01, 0.10, 0.01,
        experimental_config.rmvpe_threshold, fmt=".2f",
    )
    win.exp_rmvpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "rmvpe_threshold",
            _sl_value_as_float(win.exp_rmvpe_threshold_slider),
        )
    )
    g.addWidget(QLabel("RMVPE 阈值"), r, 0)
    g.addWidget(win.exp_rmvpe_threshold_slider, r, 1)
    g.addWidget(win.exp_rmvpe_threshold_label, r, 2); r += 1

    # ── 2. FCPE 阈值 ──
    win.exp_fcpe_threshold_slider = _slrow(
        win, "exp_fcpe_threshold_slider", 0.01, 0.10, 0.01,
        experimental_config.fcpe_confidence_threshold, fmt=".2f",
    )
    win.exp_fcpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "fcpe_confidence_threshold",
            _sl_value_as_float(win.exp_fcpe_threshold_slider),
        )
    )
    g.addWidget(QLabel("FCPE 阈值"), r, 0)
    g.addWidget(win.exp_fcpe_threshold_slider, r, 1)
    g.addWidget(win.exp_fcpe_threshold_label, r, 2); r += 1

    # ── 3. 过渡区域（双滑块：下限=中心-宽度/2，上限=中心+宽度/2） ──
    _center = experimental_config.protect_soft_threshold_hz
    _width = experimental_config.protect_soft_width
    win.exp_protect_transition_range = _range_row(
        win, "exp_protect_transition_range",
        0.0, 100.0, 1.0,
        max(0.0, _center - _width / 2), min(100.0, _center + _width / 2),
        fmt=".0f", unit="Hz",
    )
    def _on_transition_change(low, high):
        experimental_config.protect_soft_threshold_hz = (low + high) / 2
        experimental_config.protect_soft_width = high - low
    win.exp_protect_transition_range.rangeChanged.connect(_on_transition_change)
    g.addWidget(QLabel("过渡区域"), r, 0)
    g.addWidget(win.exp_protect_transition_range, r, 1)
    g.addWidget(win.exp_protect_transition_label, r, 2); r += 1

    # ── 4. 辅音保护强度 ──
    win.protect_slider = _slrow(win, "protect_slider", 0.0, 1.0, 0.01, 0.5)
    g.addWidget(QLabel("辅音保护强度"), r, 0)
    g.addWidget(win.protect_slider, r, 1)
    g.addWidget(win.protect_label, r, 2); r += 1

    # ── 5. 原声音域 ──
    win.exp_pitch_map_src_range = _range_row(
        win, "exp_pitch_map_src_range",
        20.0, 1000.0, 10.0,
        experimental_config.pitch_map_src_min, experimental_config.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    win.exp_pitch_map_src_range.rangeChanged.connect(
        lambda low, high: (
            setattr(experimental_config, "pitch_map_src_min", low),
            setattr(experimental_config, "pitch_map_src_max", high),
        )
    )
    g.addWidget(QLabel("原声音域"), r, 0)
    g.addWidget(win.exp_pitch_map_src_range, r, 1)
    g.addWidget(win.exp_pitch_map_src_label, r, 2); r += 1

    # ── 6. 目标音域 ──
    win.exp_pitch_map_dst_range = _range_row(
        win, "exp_pitch_map_dst_range",
        20.0, 1000.0, 10.0,
        experimental_config.pitch_map_dst_min, experimental_config.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    win.exp_pitch_map_dst_range.rangeChanged.connect(
        lambda low, high: (
            setattr(experimental_config, "pitch_map_dst_min", low),
            setattr(experimental_config, "pitch_map_dst_max", high),
        )
    )
    g.addWidget(QLabel("目标音域"), r, 0)
    g.addWidget(win.exp_pitch_map_dst_range, r, 1)
    g.addWidget(win.exp_pitch_map_dst_label, r, 2); r += 1

    return w
