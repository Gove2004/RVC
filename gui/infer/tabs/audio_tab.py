"""效果 Tab — 混响 / 谱减法降噪 / 3段EQ（音效处理），每项独立启用框"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QComboBox,
)

from rvc.audio import PRESETS
from gui.infer.widgets import _sl


def build_audio_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(3)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 混响 ──
    win.reverb_enable_checkbox = QCheckBox("空间混响")
    win.reverb_slider = _sl(0, 100, 1, 0)
    win.reverb_label = QLabel("0.00"); win.reverb_label.setMinimumWidth(35)
    win.reverb_slider.valueChanged.connect(lambda v: win.reverb_label.setText(f"{v / 100:.2f}"))
    g.addWidget(win.reverb_enable_checkbox, r, 0)
    g.addWidget(win.reverb_slider, r, 1)
    g.addWidget(win.reverb_label, r, 2)
    r += 1

    # ── 谱减法降噪 ──
    win.nr_enable_checkbox = QCheckBox("谱减降噪")
    win.nr_strength_slider = _sl(0, 100, 1, 50)
    win.nr_strength_label = QLabel("0.50"); win.nr_strength_label.setMinimumWidth(35)
    win.nr_strength_slider.valueChanged.connect(lambda v: win.nr_strength_label.setText(f"{v / 100:.2f}"))
    g.addWidget(win.nr_enable_checkbox, r, 0)
    g.addWidget(win.nr_strength_slider, r, 1)
    g.addWidget(win.nr_strength_label, r, 2)
    r += 1

    # ── 音效处理（3段EQ） ──
    win.eq_enable_checkbox = QCheckBox("音效处理")
    g.addWidget(win.eq_enable_checkbox, r, 0)
    win.preset_combo = QComboBox()
    win.preset_combo.addItems(PRESETS.keys())
    win.preset_combo.setFixedWidth(60)
    win.preset_combo.currentTextChanged.connect(win._apply_preset)
    g.addWidget(win.preset_combo, r, 1)
    r += 1

    win.eq_low_slider = _sl(-3000, 2000, 500, 0)
    win.eq_low_label = QLabel("0.0"); win.eq_low_label.setMinimumWidth(35)
    win.eq_low_slider.valueChanged.connect(lambda v: win.eq_low_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("低频 (200Hz)"), r, 0); g.addWidget(win.eq_low_slider, r, 1); g.addWidget(win.eq_low_label, r, 2); r += 1

    win.eq_mid_slider = _sl(-2000, 2000, 500, 0)
    win.eq_mid_label = QLabel("0.0"); win.eq_mid_label.setMinimumWidth(35)
    win.eq_mid_slider.valueChanged.connect(lambda v: win.eq_mid_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("中频 (1kHz)"), r, 0); g.addWidget(win.eq_mid_slider, r, 1); g.addWidget(win.eq_mid_label, r, 2); r += 1

    win.eq_high_slider = _sl(-3000, 3000, 500, 0)
    win.eq_high_label = QLabel("0.0"); win.eq_high_label.setMinimumWidth(35)
    win.eq_high_slider.valueChanged.connect(lambda v: win.eq_high_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("高频 (8kHz)"), r, 0); g.addWidget(win.eq_high_slider, r, 1); g.addWidget(win.eq_high_label, r, 2); r += 1

    return w
