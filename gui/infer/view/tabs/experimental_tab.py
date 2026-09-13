"""实验功能 Tab — F0 阈值 / 辅音保护 / 音域映射（无分组，直接罗列）。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel

from gui.infer.view.widgets import _slrow, RangeSlider


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
    cfg = win.runtime_params  # InferenceConfig
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    # 列宽比例：标签 20%、滑动条 65%、值 15%
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 1. 音高算法阈值（根据当前选择的 F0 方法动态切换） ──
    # 统一用一个滑动条，RMVPE/FCPE 切换时自动加载对应阈值
    initial_threshold = cfg.rmvpe_threshold if cfg.f0_method == "rmvpe" else cfg.fcpe_confidence_threshold
    win.exp_f0_threshold_slider = _slrow(
        win, "exp_f0_threshold_slider", 0.01, 0.10, 0.01,
        initial_threshold, fmt=".2f",
    )
    # 阈值标签（动态显示 RMVPE/FCPE）
    win.exp_f0_threshold_name = QLabel("RMVPE 阈值" if cfg.f0_method == "rmvpe" else "FCPE 阈值")

    g.addWidget(win.exp_f0_threshold_name, r, 0)
    g.addWidget(win.exp_f0_threshold_slider, r, 1)
    g.addWidget(win.exp_f0_threshold_label, r, 2); r += 1

    # ── 2. 辅音保护 ──
    win.exp_protect_slider = _slrow(
        win, "exp_protect_slider", 0.0, 1.0, 0.05,
        cfg.protect, fmt=".2f",
    )
    g.addWidget(QLabel("辅音保护"), r, 0)
    g.addWidget(win.exp_protect_slider, r, 1)
    g.addWidget(win.exp_protect_label, r, 2); r += 1

    # ── 2b. 辅音保护阈值（F0 低于此值视为清音）──
    win.exp_protect_threshold_slider = _slrow(
        win, "exp_protect_threshold_slider", 0.0, 50.0, 5.0,
        cfg.protect_threshold_hz, fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("辅音阈值"), r, 0)
    g.addWidget(win.exp_protect_threshold_slider, r, 1)
    g.addWidget(win.exp_protect_threshold_label, r, 2); r += 1

    # ── 4. 原声音域 ──
    win.exp_pitch_map_src_range = _range_row(
        win, "exp_pitch_map_src_range",
        20.0, 1000.0, 10.0,
        cfg.pitch_map_src_min, cfg.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("原声音域"), r, 0)
    g.addWidget(win.exp_pitch_map_src_range, r, 1)
    g.addWidget(win.exp_pitch_map_src_label, r, 2); r += 1

    # ── 5. 目标音域 ──
    win.exp_pitch_map_dst_range = _range_row(
        win, "exp_pitch_map_dst_range",
        20.0, 1000.0, 10.0,
        cfg.pitch_map_dst_min, cfg.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("目标音域"), r, 0)
    g.addWidget(win.exp_pitch_map_dst_range, r, 1)
    g.addWidget(win.exp_pitch_map_dst_label, r, 2); r += 1

    return w
