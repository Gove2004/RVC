"""全局参数 Tab — 采样/融合参数 + 辅音保护 + 频谱降噪"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox,
)

from gui.infer.widgets import _sl


def build_global_params_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    r = 0

    # ── 采样与融合参数 ──
    win.block_time_slider = _sl(5, 100, 1, 25)
    win.block_time_label = QLabel("0.25"); win.block_time_label.setMinimumWidth(35)
    win.block_time_slider.valueChanged.connect(lambda v: win.block_time_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("采样长度"), r, 0); g.addWidget(win.block_time_slider, r, 1); g.addWidget(win.block_time_label, r, 2); r += 1

    win.crossfade_slider = _sl(1, 15, 1, 5)
    win.crossfade_label = QLabel("0.05"); win.crossfade_label.setMinimumWidth(35)
    win.crossfade_slider.valueChanged.connect(lambda v: win.crossfade_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("淡入长度"), r, 0); g.addWidget(win.crossfade_slider, r, 1); g.addWidget(win.crossfade_label, r, 2); r += 1

    win.extra_time_slider = _sl(5, 500, 1, 250)
    win.extra_time_label = QLabel("2.50"); win.extra_time_label.setMinimumWidth(35)
    win.extra_time_slider.valueChanged.connect(lambda v: win.extra_time_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("额外上下文"), r, 0); g.addWidget(win.extra_time_slider, r, 1); g.addWidget(win.extra_time_label, r, 2); r += 1

    win.rms_mix_slider = _sl(0, 100, 1, 0)
    win.rms_mix_label = QLabel("0.00"); win.rms_mix_label.setMinimumWidth(35)
    win.rms_mix_slider.valueChanged.connect(lambda v: win.rms_mix_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("响度因子"), r, 0); g.addWidget(win.rms_mix_slider, r, 1); g.addWidget(win.rms_mix_label, r, 2); r += 1

    # ── 辅音保护 ──
    win.protect_slider = _sl(0, 100, 1, 25)
    win.protect_label = QLabel("0.25"); win.protect_label.setMinimumWidth(35)
    win.protect_slider.valueChanged.connect(lambda v: win.protect_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("辅音保护"), r, 0); g.addWidget(win.protect_slider, r, 1); g.addWidget(win.protect_label, r, 2); r += 1

    # ── 频谱降噪（输入侧 GPU 谱减法） ──
    win.nr_enable_checkbox = QCheckBox("频谱降噪")
    g.addWidget(win.nr_enable_checkbox, r, 0)
    win.nr_strength_slider = _sl(0, 100, 1, 50)
    win.nr_strength_label = QLabel("0.50"); win.nr_strength_label.setMinimumWidth(35)
    win.nr_strength_slider.valueChanged.connect(lambda v: win.nr_strength_label.setText(f"{v / 100:.2f}"))
    g.addWidget(win.nr_strength_slider, r, 1); g.addWidget(win.nr_strength_label, r, 2); r += 1

    return w
