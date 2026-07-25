"""声学 Tab — 音效（5段EQ + 混响 + 预设系统）"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QComboBox,
    QHBoxLayout, QVBoxLayout,
)
from rvc.audio import PRESETS
from gui.infer.widgets import _sl
from gui.styles import sep



def build_audio_tab(win):
    w = QWidget()
    root = QVBoxLayout(w); root.setSpacing(4); root.setContentsMargins(8, 8, 8, 8)

    row0 = QHBoxLayout()
    win.eq_enable_checkbox = QCheckBox("开启音效")
    win.preset_combo = QComboBox(); win.preset_combo.addItems(PRESETS.keys()); win.preset_combo.setFixedWidth(60)
    win.preset_combo.currentTextChanged.connect(win._apply_preset)
    row0.addWidget(win.eq_enable_checkbox); row0.addWidget(QLabel("预设:")); row0.addWidget(win.preset_combo); row0.addStretch()
    gw = QWidget(); gw.setLayout(row0); root.addWidget(gw)

    root.addWidget(sep())

    g = QGridLayout(); g.setSpacing(4)
    gw2 = QWidget(); gw2.setLayout(g)
    root.addWidget(gw2)

    win.eq_sub_slider = _sl(-2000, 2000, 500, 0); win.eq_sub_label = QLabel("0.0"); win.eq_sub_label.setMinimumWidth(35)
    win.eq_sub_slider.valueChanged.connect(lambda v: win.eq_sub_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("超低频 (60Hz)"), 0, 0); g.addWidget(win.eq_sub_slider, 0, 1); g.addWidget(win.eq_sub_label, 0, 2)

    win.eq_low_slider = _sl(-3000, 2000, 500, 0); win.eq_low_label = QLabel("0.0"); win.eq_low_label.setMinimumWidth(35)
    win.eq_low_slider.valueChanged.connect(lambda v: win.eq_low_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("低频 (200Hz)"), 1, 0); g.addWidget(win.eq_low_slider, 1, 1); g.addWidget(win.eq_low_label, 1, 2)

    win.eq_mid_slider = _sl(-2000, 2000, 500, 0); win.eq_mid_label = QLabel("0.0"); win.eq_mid_label.setMinimumWidth(35)
    win.eq_mid_slider.valueChanged.connect(lambda v: win.eq_mid_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("中频 (1kHz)"), 2, 0); g.addWidget(win.eq_mid_slider, 2, 1); g.addWidget(win.eq_mid_label, 2, 2)

    win.eq_hi_mid_slider = _sl(-2000, 2000, 500, 0); win.eq_hi_mid_label = QLabel("0.0"); win.eq_hi_mid_label.setMinimumWidth(35)
    win.eq_hi_mid_slider.valueChanged.connect(lambda v: win.eq_hi_mid_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("中高频 (3kHz)"), 3, 0); g.addWidget(win.eq_hi_mid_slider, 3, 1); g.addWidget(win.eq_hi_mid_label, 3, 2)

    win.eq_high_slider = _sl(-3000, 3000, 500, 0); win.eq_high_label = QLabel("0.0"); win.eq_high_label.setMinimumWidth(35)
    win.eq_high_slider.valueChanged.connect(lambda v: win.eq_high_label.setText(f"{v / 100:.1f}"))
    g.addWidget(QLabel("高频 (8kHz)"), 4, 0); g.addWidget(win.eq_high_slider, 4, 1); g.addWidget(win.eq_high_label, 4, 2)

    root.addWidget(sep())

    reverb_row = QHBoxLayout()
    win.reverb_slider = _sl(0, 50, 1, 0); win.reverb_label = QLabel("0.00"); win.reverb_label.setMinimumWidth(35)
    win.reverb_slider.valueChanged.connect(lambda v: win.reverb_label.setText(f"{v / 100:.2f}"))
    reverb_row.addWidget(QLabel("空间混响")); reverb_row.addWidget(win.reverb_slider); reverb_row.addWidget(win.reverb_label); reverb_row.addStretch()
    rv = QWidget(); rv.setLayout(reverb_row)
    root.addWidget(rv)

    return w
