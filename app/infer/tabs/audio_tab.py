"""声学 Tab — 降噪 + 音效（EQ/饱和/压限/混响 + 预设系统）"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QComboBox,
    QHBoxLayout,
)
from PySide6.QtCore import Qt
from rvc.audio_utils import PRESETS
from app.infer.widgets import _sl
from app.infer.tabs.settings_tab import sep


def build_audio_tab(win):
    """构建「声学」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
    g.setColumnStretch(1, 1)
    r = 0

    # 降噪
    win.inr = QCheckBox("输入降噪"); win.ounr = QCheckBox("输出降噪")
    nr = QHBoxLayout(); nr.addWidget(win.inr); nr.addWidget(win.ounr); nr.addStretch()
    g.addLayout(nr, r, 0, 1, 3); r+=1
    g.addWidget(sep(), r, 0, 1, 3); r+=1

    # 音效
    win.eq_en = QCheckBox("开启音效")
    win.preset_combo = QComboBox(); win.preset_combo.addItems(PRESETS.keys()); win.preset_combo.setFixedWidth(90)
    win.preset_combo.currentTextChanged.connect(win._apply_preset)
    row0 = QHBoxLayout(); row0.addWidget(win.eq_en); row0.addWidget(win.preset_combo); row0.addStretch()
    g.addLayout(row0, r, 0, 1, 3); r+=1

    def add_eq(label, sl, lbl, row):
        g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

    win.eq_lo = _sl(-3000,2000,500,0); win.eq_lo_lbl = QLabel("0.0"); win.eq_lo_lbl.setMinimumWidth(35)
    win.eq_lo.valueChanged.connect(lambda v: win.eq_lo_lbl.setText(f"{v/100:.1f}"))
    add_eq("低频增益", win.eq_lo, win.eq_lo_lbl, r); r+=1
    win.eq_mi = _sl(-2000,2000,500,0); win.eq_mi_lbl = QLabel("0.0"); win.eq_mi_lbl.setMinimumWidth(35)
    win.eq_mi.valueChanged.connect(lambda v: win.eq_mi_lbl.setText(f"{v/100:.1f}"))
    add_eq("中频增益", win.eq_mi, win.eq_mi_lbl, r); r+=1
    win.eq_hi = _sl(-3000,3000,500,0); win.eq_hi_lbl = QLabel("0.0"); win.eq_hi_lbl.setMinimumWidth(35)
    win.eq_hi.valueChanged.connect(lambda v: win.eq_hi_lbl.setText(f"{v/100:.1f}"))
    add_eq("高频增益", win.eq_hi, win.eq_hi_lbl, r); r+=1
    win.warm_sl = _sl(0,100,1,0); win.warm_lbl = QLabel("0.00"); win.warm_lbl.setMinimumWidth(35)
    win.warm_sl.valueChanged.connect(lambda v: win.warm_lbl.setText(f"{v/100:.2f}"))
    add_eq("电子管饱和", win.warm_sl, win.warm_lbl, r); r+=1
    win.comp_sl = _sl(0,100,5,0); win.comp_lbl = QLabel("0.00"); win.comp_lbl.setMinimumWidth(35)
    win.comp_sl.valueChanged.connect(lambda v: win.comp_lbl.setText(f"{v/100:.2f}"))
    add_eq("动态压限", win.comp_sl, win.comp_lbl, r); r+=1
    win.rev_sl = _sl(0,50,1,0); win.rev_lbl = QLabel("0.00"); win.rev_lbl.setMinimumWidth(35)
    win.rev_sl.valueChanged.connect(lambda v: win.rev_lbl.setText(f"{v/100:.2f}"))
    add_eq("空间混响", win.rev_sl, win.rev_lbl, r)
    return w
