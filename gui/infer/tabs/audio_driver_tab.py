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

    # ── 采样率 ──（QButtonGroup 隔离，避免与下方音高算法互斥）
    win.sr_model_radio = QRadioButton("模型 -")
    win.sr_model_radio.setMaximumWidth(100)
    win.sr_device_radio = QRadioButton("设备 -")
    win.sr_device_radio.setMaximumWidth(100)
    win.sr_model_radio.setChecked(True)

    sr_group = QButtonGroup(w)
    sr_group.addButton(win.sr_model_radio)
    sr_group.addButton(win.sr_device_radio)

    refresh_btn = QPushButton("刷新")
    refresh_btn.setStyleSheet(ButtonStyles.secondary())
    # 连接将在 window 中进行

    g.addWidget(refresh_btn, r, 0)
    g.addWidget(win.sr_model_radio, r, 1)
    g.addWidget(win.sr_device_radio, r, 2)
    r += 1

    # ── 音高算法 ──（独立 QButtonGroup，与采样率组互不干扰）
    win.f0_rmvp_btn = QRadioButton("RMVPE")
    win.f0_rmvp_btn.setMaximumWidth(80)
    win.f0_fcpe_btn = QRadioButton("FCPE")
    win.f0_fcpe_btn.setMaximumWidth(80)
    win.f0_fcpe_btn.setChecked(True)  # 默认 FCPE

    f0_group = QButtonGroup(w)
    f0_group.addButton(win.f0_rmvp_btn)
    f0_group.addButton(win.f0_fcpe_btn)

    g.addWidget(QLabel("音高算法"), r, 0)
    g.addWidget(win.f0_rmvp_btn, r, 1)
    g.addWidget(win.f0_fcpe_btn, r, 2)
    r += 1

    return w, refresh_btn
