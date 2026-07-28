"""音频驱动 Tab — 音频设备 + 采样率显示"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox,
    QPushButton, QRadioButton, QButtonGroup,
)
from PySide6.QtCore import Qt

from gui.styles import ButtonStyles, LabelStyles
from gui.infer.widgets import _sl


def build_audio_driver_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    r = 0

    # ── 设备选择 ──
    win.hostapi_combo = QComboBox()
    win.hostapi_combo.currentTextChanged.connect(win._ha_changed)
    g.addWidget(QLabel("音频驱动"), r, 0)
    g.addWidget(win.hostapi_combo, r, 1, 1, 2)
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

    # ── 采样率 ──（单行排列，和上面一致）
    win.sr_model_radio = QRadioButton("模型 -")
    win.sr_model_radio.setMaximumWidth(50)
    win.sr_device_radio = QRadioButton("设备 -")
    win.sr_device_radio.setMaximumWidth(50)
    win.sr_model_radio.setChecked(True)

    refresh_btn = QPushButton("刷新")
    refresh_btn.setStyleSheet(ButtonStyles.secondary())
    # 连接将在 window 中进行

    g.addWidget(refresh_btn, r, 0)
    g.addWidget(win.sr_model_radio, r, 1)
    g.addWidget(win.sr_device_radio, r, 2)

    return w, refresh_btn