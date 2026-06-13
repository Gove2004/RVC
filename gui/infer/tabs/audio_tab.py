"""声学 Tab — 音效（5段EQ + 混响 + 预设系统）"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QComboBox,
    QHBoxLayout,
)
from PySide6.QtCore import Qt
from rvc.audio import PRESETS
from gui.infer.widgets import _sl
from gui.infer.tabs.settings_tab import sep


def build_audio_tab(win):
    """构建「声学」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(8,8,8,8)
    g.setColumnStretch(1, 1)
    r = 0

    # 音效总开关和预设
    win.eq_enable_checkbox = QCheckBox("开启音效")
    win.preset_combo = QComboBox(); win.preset_combo.addItems(PRESETS.keys()); win.preset_combo.setFixedWidth(60)
    win.preset_combo.currentTextChanged.connect(win._apply_preset)
    row0 = QHBoxLayout(); row0.addWidget(win.eq_enable_checkbox); row0.addWidget(win.preset_combo); row0.addStretch()
    g.addLayout(row0, r, 0, 1, 3); r+=1

    def add_eq_slider(label, slider, value_label, row):
        g.addWidget(QLabel(label), row, 0); g.addWidget(slider, row, 1); g.addWidget(value_label, row, 2)

    # 5段EQ：超低频、低频、中频、中高频、高频
    win.eq_sub_slider = _sl(-2000,2000,500,0); win.eq_sub_label = QLabel("0.0"); win.eq_sub_label.setMinimumWidth(35)
    win.eq_sub_slider.valueChanged.connect(lambda v: win.eq_sub_label.setText(f"{v/100:.1f}"))
    add_eq_slider("超低频 (60Hz)", win.eq_sub_slider, win.eq_sub_label, r); r+=1

    win.eq_low_slider = _sl(-3000,2000,500,0); win.eq_low_label = QLabel("0.0"); win.eq_low_label.setMinimumWidth(35)
    win.eq_low_slider.valueChanged.connect(lambda v: win.eq_low_label.setText(f"{v/100:.1f}"))
    add_eq_slider("低频 (200Hz)", win.eq_low_slider, win.eq_low_label, r); r+=1

    win.eq_mid_slider = _sl(-2000,2000,500,0); win.eq_mid_label = QLabel("0.0"); win.eq_mid_label.setMinimumWidth(35)
    win.eq_mid_slider.valueChanged.connect(lambda v: win.eq_mid_label.setText(f"{v/100:.1f}"))
    add_eq_slider("中频 (1kHz)", win.eq_mid_slider, win.eq_mid_label, r); r+=1

    win.eq_hi_mid_slider = _sl(-2000,2000,500,0); win.eq_hi_mid_label = QLabel("0.0"); win.eq_hi_mid_label.setMinimumWidth(35)
    win.eq_hi_mid_slider.valueChanged.connect(lambda v: win.eq_hi_mid_label.setText(f"{v/100:.1f}"))
    add_eq_slider("中高频 (3kHz)", win.eq_hi_mid_slider, win.eq_hi_mid_label, r); r+=1

    win.eq_high_slider = _sl(-3000,3000,500,0); win.eq_high_label = QLabel("0.0"); win.eq_high_label.setMinimumWidth(35)
    win.eq_high_slider.valueChanged.connect(lambda v: win.eq_high_label.setText(f"{v/100:.1f}"))
    add_eq_slider("高频 (8kHz)", win.eq_high_slider, win.eq_high_label, r); r+=1

    # 混响
    win.reverb_slider = _sl(0,50,1,0); win.reverb_label = QLabel("0.00"); win.reverb_label.setMinimumWidth(35)
    win.reverb_slider.valueChanged.connect(lambda v: win.reverb_label.setText(f"{v/100:.2f}"))
    add_eq_slider("空间混响", win.reverb_slider, win.reverb_label, r)
    return w
