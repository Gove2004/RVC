"""基础设置 Tab — 音频设备 + 引擎参数"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox, QSlider,
    QHBoxLayout, QPushButton, QRadioButton, QButtonGroup,
)
from PySide6.QtCore import Qt
from gui.infer.widgets import _sl
from gui.styles import ButtonStyles, Colors, LabelStyles, sep


def build_settings_tab(win):
    w = QWidget(); g = QGridLayout(w); g.setSpacing(4); g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 音频驱动 ──
    win.hostapi_combo = QComboBox(); win.hostapi_combo.currentTextChanged.connect(win._ha_changed)
    g.addWidget(QLabel("音频驱动"), r, 0); g.addWidget(win.hostapi_combo, r, 1, 1, 2); r += 1

    # ── 麦克风 ──
    win.input_combo = QComboBox()
    g.addWidget(QLabel("麦克风"), r, 0); g.addWidget(win.input_combo, r, 1, 1, 2); r += 1

    # ── 主输出 ──
    win.output_combo = QComboBox()
    g.addWidget(QLabel("主输出"), r, 0); g.addWidget(win.output_combo, r, 1, 1, 2); r += 1

    # ── 副输出 ──
    win.output2_combo = QComboBox()
    g.addWidget(QLabel("副输出"), r, 0); g.addWidget(win.output2_combo, r, 1, 1, 2); r += 1

    g.addWidget(sep(), r, 0, 1, 3); r += 1

    win.block_time_slider = _sl(2, 150, 1, 25)
    win.block_time_label = QLabel("0.25"); win.block_time_label.setMinimumWidth(35)
    win.block_time_slider.valueChanged.connect(lambda v: win.block_time_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("采样长度"), r, 0); g.addWidget(win.block_time_slider, r, 1); g.addWidget(win.block_time_label, r, 2); r += 1

    win.crossfade_slider = _sl(1, 15, 1, 5)
    win.crossfade_label = QLabel("0.05"); win.crossfade_label.setMinimumWidth(35)
    win.crossfade_slider.valueChanged.connect(lambda v: win.crossfade_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("淡入淡出"), r, 0); g.addWidget(win.crossfade_slider, r, 1); g.addWidget(win.crossfade_label, r, 2); r += 1

    win.extra_time_slider = _sl(5, 500, 1, 250)
    win.extra_time_label = QLabel("2.50"); win.extra_time_label.setMinimumWidth(35)
    win.extra_time_slider.valueChanged.connect(lambda v: win.extra_time_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("额外上下文"), r, 0); g.addWidget(win.extra_time_slider, r, 1); g.addWidget(win.extra_time_label, r, 2); r += 1

    win.rms_mix_slider = _sl(0, 100, 1, 0)
    win.rms_mix_label = QLabel("0.00"); win.rms_mix_label.setMinimumWidth(35)
    win.rms_mix_slider.valueChanged.connect(lambda v: win.rms_mix_label.setText(f"{v / 100:.2f}"))
    g.addWidget(QLabel("响度因子"), r, 0); g.addWidget(win.rms_mix_slider, r, 1); g.addWidget(win.rms_mix_label, r, 2); r += 1

    g.addWidget(sep(), r, 0, 1, 3); r += 1

    # ── 采样率 ──
    win.sr_model_radio = QRadioButton("模型"); win.sr_model_radio.setChecked(True)
    win.sr_device_radio = QRadioButton("设备")
    sr_group = QButtonGroup(w); sr_group.addButton(win.sr_model_radio); sr_group.addButton(win.sr_device_radio)
    win.sr_model_value = QLabel("-"); win.sr_model_value.setStyleSheet(LabelStyles.status())
    win.sr_device_value = QLabel("-"); win.sr_device_value.setStyleSheet(LabelStyles.status())

    sr_w = QWidget(); srr = QHBoxLayout(sr_w); srr.setSpacing(6); srr.setContentsMargins(0, 0, 0, 0)
    srr.addWidget(QLabel("采样率")); srr.addWidget(win.sr_model_radio); srr.addWidget(win.sr_model_value)
    srr.addSpacing(12)
    srr.addWidget(win.sr_device_radio); srr.addWidget(win.sr_device_value); srr.addStretch()
    g.addWidget(sr_w, r, 0, 1, 3); r += 1

    # ── 音高算法 ──
    win.f0_rmvp_btn = QRadioButton("RMVPE"); win.f0_fcpe_btn = QRadioButton("FCPE")
    f0_group = QButtonGroup(w); f0_group.addButton(win.f0_rmvp_btn); f0_group.addButton(win.f0_fcpe_btn)
    f0_method = "fcpe"
    if f0_method == "rmvpe":
        win.f0_rmvp_btn.setChecked(True)
    else:
        win.f0_fcpe_btn.setChecked(True)

    f0_w = QWidget(); f0r = QHBoxLayout(f0_w); f0r.setSpacing(6); f0r.setContentsMargins(0, 0, 0, 0)
    f0r.addWidget(QLabel("音高算法")); f0r.addWidget(win.f0_rmvp_btn); f0r.addWidget(win.f0_fcpe_btn); f0r.addStretch()
    g.addWidget(f0_w, r, 0, 1, 3)

    return w
