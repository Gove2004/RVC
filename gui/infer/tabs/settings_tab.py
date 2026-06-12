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
    w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
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
    br = QPushButton("刷新"); br.setFixedWidth(45); br.clicked.connect(win._reload_dev)
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

    win.bl_sl = _sl(2,150,1,25); win.bl_lbl = QLabel("0.25"); win.bl_lbl.setMinimumWidth(35)
    win.bl_sl.valueChanged.connect(lambda v: win.bl_lbl.setText(f"{v/100:.2f}"))
    add_sl("采样长度", win.bl_sl, win.bl_lbl, r); r+=1
    win.cf_sl = _sl(1,15,1,5); win.cf_lbl = QLabel("0.05"); win.cf_lbl.setMinimumWidth(35)
    win.cf_sl.valueChanged.connect(lambda v: win.cf_lbl.setText(f"{v/100:.2f}"))
    add_sl("淡入淡出", win.cf_sl, win.cf_lbl, r); r+=1
    win.ex_sl = _sl(5,500,1,250); win.ex_lbl = QLabel("2.50"); win.ex_lbl.setMinimumWidth(35)
    win.ex_sl.valueChanged.connect(lambda v: win.ex_lbl.setText(f"{v/100:.2f}"))
    add_sl("额外上下文", win.ex_sl, win.ex_lbl, r); r+=1

    g.addWidget(QLabel("音高算法"), r, 0)
    win.f0_combo = QComboBox(); win.f0_combo.addItems(["fcpe", "rmvpe"]); win.f0_combo.setFixedWidth(80)
    g.addWidget(win.f0_combo, r, 1)
    return w
