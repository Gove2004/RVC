"""设置 Tab — 音频设备路由 + 引擎参数"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox, QSlider,
    QRadioButton, QHBoxLayout, QPushButton, QFrame,
)
from PySide6.QtCore import Qt
from gui.infer.widgets import _sl
from gui.styles import ButtonStyles, Colors


def sep():
    """水平分隔线"""
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet(f"color:{Colors.DIVIDER}"); return f


def build_settings_tab(win):
    """构建「设置」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(8,8,8,8)
    g.setColumnStretch(1, 1)
    r = 0

    g.addWidget(QLabel("音频驱动"), r, 0)
    win.hostapi_combo = QComboBox(); win.hostapi_combo.currentTextChanged.connect(win._ha_changed)
    g.addWidget(win.hostapi_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("麦克风"), r, 0)
    win.input_combo = QComboBox(); g.addWidget(win.input_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("主输出"), r, 0)
    win.output_combo = QComboBox(); g.addWidget(win.output_combo, r, 1, 1, 2); r+=1
    g.addWidget(QLabel("副输出"), r, 0)
    win.output2_combo = QComboBox(); g.addWidget(win.output2_combo, r, 1)
    refresh_btn = QPushButton("刷新"); refresh_btn.setFixedWidth(30)
    refresh_btn.setStyleSheet(ButtonStyles.small())
    refresh_btn.clicked.connect(win._reload_dev)
    g.addWidget(refresh_btn, r, 2); r+=1
    win.sr_model_radio = QRadioButton(); win.sr_model_radio.setChecked(True)
    win.sr_device_radio = QRadioButton()
    win.sr_model_label = QLabel("模型采样率: -")
    win.sr_device_label = QLabel("设备采样率: -")
    sr = QHBoxLayout(); sr.setSpacing(4)
    sr.addWidget(win.sr_model_radio); sr.addWidget(win.sr_model_label)
    sr.addSpacing(12); sr.addWidget(win.sr_device_radio); sr.addWidget(win.sr_device_label); sr.addStretch()
    g.addLayout(sr, r, 0, 1, 3); r+=1

    g.addWidget(sep(), r, 0, 1, 3); r+=1

    def add_slider(label, slider, value_label, row):
        g.addWidget(QLabel(label), row, 0); g.addWidget(slider, row, 1); g.addWidget(value_label, row, 2)

    # 采样长度 (block time)
    win.block_time_slider = _sl(2,150,1,25); win.block_time_label = QLabel("0.25"); win.block_time_label.setMinimumWidth(35)
    win.block_time_slider.valueChanged.connect(lambda v: win.block_time_label.setText(f"{v/100:.2f}"))
    add_slider("采样长度", win.block_time_slider, win.block_time_label, r); r+=1

    # 淡入淡出 (crossfade)
    win.crossfade_slider = _sl(1,15,1,5); win.crossfade_label = QLabel("0.05"); win.crossfade_label.setMinimumWidth(35)
    win.crossfade_slider.valueChanged.connect(lambda v: win.crossfade_label.setText(f"{v/100:.2f}"))
    add_slider("淡入淡出", win.crossfade_slider, win.crossfade_label, r); r+=1

    # 额外上下文 (extra context)
    win.extra_time_slider = _sl(5,500,1,250); win.extra_time_label = QLabel("2.50"); win.extra_time_label.setMinimumWidth(35)
    win.extra_time_slider.valueChanged.connect(lambda v: win.extra_time_label.setText(f"{v/100:.2f}"))
    add_slider("额外上下文", win.extra_time_slider, win.extra_time_label, r); r+=1

    g.addWidget(QLabel("音高算法"), r, 0)
    win.f0_combo = QComboBox(); win.f0_combo.addItems(["fcpe", "rmvpe"]); win.f0_combo.setFixedWidth(53)
    g.addWidget(win.f0_combo, r, 1)
    return w
