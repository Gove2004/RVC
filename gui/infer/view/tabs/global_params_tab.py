"""全局参数 Tab — 采样/融合参数"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox,
)

from gui.infer.view.widgets import _slrow


def build_global_params_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    r = 0

    # ── 采样与融合参数 ──
    win.block_time_slider = _slrow(win, "block_time_slider", 0.05, 1.0, 0.01, 0.25)
    g.addWidget(QLabel("采样长度"), r, 0); g.addWidget(win.block_time_slider, r, 1); g.addWidget(win.block_time_label, r, 2); r += 1

    win.crossfade_slider = _slrow(win, "crossfade_slider", 0.01, 0.15, 0.01, 0.05)
    g.addWidget(QLabel("淡入长度"), r, 0); g.addWidget(win.crossfade_slider, r, 1); g.addWidget(win.crossfade_label, r, 2); r += 1

    win.extra_time_slider = _slrow(win, "extra_time_slider", 0.05, 5.0, 0.01, 2.5)
    g.addWidget(QLabel("额外上下文"), r, 0); g.addWidget(win.extra_time_slider, r, 1); g.addWidget(win.extra_time_label, r, 2); r += 1

    win.rms_mix_slider = _slrow(win, "rms_mix_slider", 0.0, 1.0, 0.01, 0.0)
    g.addWidget(QLabel("响度因子"), r, 0); g.addWidget(win.rms_mix_slider, r, 1); g.addWidget(win.rms_mix_label, r, 2); r += 1

    return w
