"""音频驱动 Tab — 音频设备 + 采样率"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox,
    QPushButton, QRadioButton, QButtonGroup,
)

from gui.styles import ButtonStyles


def build_audio_driver_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 设备选择 ──
    win.hostapi_combo = QComboBox()
    win.hostapi_combo.currentTextChanged.connect(win._ha_changed)
    refresh_btn = QPushButton("刷新")
    refresh_btn.setStyleSheet(ButtonStyles.secondary())
    g.addWidget(QLabel("音频驱动"), r, 0)
    g.addWidget(win.hostapi_combo, r, 1)
    g.addWidget(refresh_btn, r, 2)
    r += 1

    win.input_combo = QComboBox()
    g.addWidget(QLabel("麦克风"), r, 0)
    g.addWidget(win.input_combo, r, 1, 1, 2)
    r += 1

    win.output_combo = QComboBox()
    g.addWidget(QLabel("主输出"), r, 0)
    g.addWidget(win.output_combo, r, 1, 1, 2)
    r += 1

    win.output2_combo = QComboBox()
    g.addWidget(QLabel("副输出"), r, 0)
    g.addWidget(win.output2_combo, r, 1, 1, 2)
    r += 1

    # ── 采样率 ──
    win.sr_model_radio = QRadioButton("模型 -")
    win.sr_model_radio.setMaximumWidth(100)
    win.sr_device_radio = QRadioButton("设备 -")
    win.sr_device_radio.setMaximumWidth(100)
    win.sr_model_radio.setChecked(True)

    sr_group = QButtonGroup(w)
    sr_group.addButton(win.sr_model_radio)
    sr_group.addButton(win.sr_device_radio)

    g.addWidget(QLabel("采样率"), r, 0)
    g.addWidget(win.sr_model_radio, r, 1)
    g.addWidget(win.sr_device_radio, r, 2)
    r += 1

    return w, refresh_btn
