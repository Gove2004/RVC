"""全局参数 Tab — 采样/融合参数 + 音高算法 + 辅音保护"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox, QSlider,
    QRadioButton, QButtonGroup,
    QHBoxLayout,
)
from PySide6.QtCore import Qt

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

    # ── 音高算法 ──（单行排列，和上面一致）
    win.f0_rmvp_btn = QRadioButton("RMVPE")
    win.f0_rmvp_btn.setMaximumWidth(80)   # 从 50 → 80
    win.f0_fcpe_btn = QRadioButton("FCPE")
    win.f0_fcpe_btn.setMaximumWidth(80)   # 从 50 → 80
    win.f0_fcpe_btn.setChecked(True)  # 默认 FCPE

    g.addWidget(QLabel("音高算法"), r, 0)
    g.addWidget(win.f0_rmvp_btn, r, 1)
    g.addWidget(win.f0_fcpe_btn, r, 2)

    return w