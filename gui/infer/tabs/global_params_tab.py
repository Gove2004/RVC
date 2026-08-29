"""全局参数 Tab — 采样/融合参数 + 辅音保护 + 频谱降噪 + 破音保护"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox,
)

from gui.infer.widgets import _slrow


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

    # ── 辅音保护 ──
    win.protect_slider = _slrow(win, "protect_slider", 0.0, 1.0, 0.01, 0.5)
    g.addWidget(QLabel("辅音保护"), r, 0); g.addWidget(win.protect_slider, r, 1); g.addWidget(win.protect_label, r, 2); r += 1

    # ── 频谱降噪（输入侧 GPU 谱减法） ──
    win.nr_enable_checkbox = QCheckBox("频谱降噪")
    g.addWidget(win.nr_enable_checkbox, r, 0)
    win.nr_strength_slider = _slrow(win, "nr_strength_slider", 0.0, 1.0, 0.01, 0.5)
    g.addWidget(win.nr_strength_slider, r, 1); g.addWidget(win.nr_strength_label, r, 2); r += 1

    # ── 破音保护（核心：高音破音/沙哑。源音高超过临界收敛） ──
    win.break_enable_checkbox = QCheckBox("破音保护")
    g.addWidget(win.break_enable_checkbox, r, 0)
    win.break_src_hz_slider = _slrow(win, "break_src_hz_slider", 200.0, 400.0, 5.0, 300.0, fmt=".0f", unit="Hz", label_w=45)
    g.addWidget(win.break_src_hz_slider, r, 1); g.addWidget(win.break_src_hz_label, r, 2); r += 1

    return w
