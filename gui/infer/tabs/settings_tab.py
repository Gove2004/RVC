"""设置 Tab — 音频设备路由 + 引擎参数"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox, QSlider,
    QRadioButton, QHBoxLayout, QPushButton, QFrame,
)
from PySide6.QtCore import Qt
from gui.infer.widgets import _sl


def sep():
    """水平分隔线"""
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet("color:#444"); return f


def build_settings_tab(win):
    """构建「设置」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(8,8,8,8)
    g.setColumnStretch(1, 1)
    r = 0

    g.addWidget(QLabel("音频驱动"), r, 0)
    win.ha_combo = QComboBox(); win.ha_combo.currentTextChanged.connect(win._ha_changed)
    g.addWidget(win.ha_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("麦克风"), r, 0)
    win.in_combo = QComboBox(); g.addWidget(win.in_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("主输出"), r, 0)
    win.out_combo = QComboBox(); g.addWidget(win.out_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("副输出"), r, 0)
    win.out2_combo = QComboBox(); g.addWidget(win.out2_combo, r, 1)
    br = QPushButton("刷新"); br.setFixedWidth(30); br.clicked.connect(win._reload_dev)
    g.addWidget(br, r, 2); r+=1
    win.sr_r1 = QRadioButton(); win.sr_r1.setChecked(True)
    win.sr_r2 = QRadioButton()
    win.sr_r1_lbl = QLabel("模型采样率: -")
    win.sr_r2_lbl = QLabel("设备采样率: -")
    sr = QHBoxLayout(); sr.setSpacing(4)
    sr.addWidget(win.sr_r1); sr.addWidget(win.sr_r1_lbl)
    sr.addSpacing(12); sr.addWidget(win.sr_r2); sr.addWidget(win.sr_r2_lbl); sr.addStretch()
    g.addLayout(sr, r, 0, 1, 3); r+=1

    g.addWidget(sep(), r, 0, 1, 3); r+=1

    def add_sl(label, sl, lbl, row):
        g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

    # 采样长度 (block time)
    win.block_time_slider = _sl(2,150,1,25); win.block_time_label = QLabel("0.25"); win.block_time_label.setMinimumWidth(35)
    win.block_time_slider.valueChanged.connect(lambda v: win.block_time_label.setText(f"{v/100:.2f}"))
    add_sl("采样长度", win.block_time_slider, win.block_time_label, r); r+=1

    # 淡入淡出 (crossfade)
    win.crossfade_slider = _sl(1,15,1,5); win.crossfade_label = QLabel("0.05"); win.crossfade_label.setMinimumWidth(35)
    win.crossfade_slider.valueChanged.connect(lambda v: win.crossfade_label.setText(f"{v/100:.2f}"))
    add_sl("淡入淡出", win.crossfade_slider, win.crossfade_label, r); r+=1

    # 额外上下文 (extra context)
    win.extra_time_slider = _sl(5,500,1,250); win.extra_time_label = QLabel("2.50"); win.extra_time_label.setMinimumWidth(35)
    win.extra_time_slider.valueChanged.connect(lambda v: win.extra_time_label.setText(f"{v/100:.2f}"))
    add_sl("额外上下文", win.extra_time_slider, win.extra_time_label, r); r+=1

    # 保留旧变量名作为别名以兼容
    win.bl_sl = win.block_time_slider
    win.bl_lbl = win.block_time_label
    win.cf_sl = win.crossfade_slider
    win.cf_lbl = win.crossfade_label
    win.ex_sl = win.extra_time_slider
    win.ex_lbl = win.extra_time_label

    g.addWidget(QLabel("音高算法"), r, 0)
    win.f0_combo = QComboBox(); win.f0_combo.addItems(["fcpe", "rmvpe"]); win.f0_combo.setFixedWidth(53)
    g.addWidget(win.f0_combo, r, 1)
    return w
