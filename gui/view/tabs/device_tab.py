"""设备 Tab — 推理显卡 / 音频驱动与设备 / 采样率 / 停机直通"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox,
    QPushButton, QRadioButton, QButtonGroup, QCheckBox,
)

from gui.styles import ButtonStyles


def build_device_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 推理显卡（多卡机器选卡）──
    # 列表在引擎预热线程里探测后回填（torch 晚加载，见 window._populate_gpu_combo）；
    # 此处的占位项保证控件无条件存在，避免契约里出现「有时有有时无」的属性。
    win.gpu_combo = QComboBox()
    win.gpu_combo.addItem("GPU 0")
    win.gpu_combo.setToolTip(
        "推理使用的 NVIDIA GPU（多卡机器）。\n"
        "切换后需重启程序生效：模型权重与 CUDA Graph 已按当前显卡驻留。"
    )
    g.addWidget(QLabel("显卡驱动"), r, 0)
    g.addWidget(win.gpu_combo, r, 1, 1, 2)
    r += 1

    # ── 音频驱动（刷新按钮由 window 在 device_catalog 就绪后连信号）──
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

    # ── 停机直通 ──
    win.passthrough_check = QCheckBox("引擎停机期间麦克风原声直出主输出")
    win.passthrough_check.setToolTip(
        "勾选后：点「停止」（或运行出错停机）时自动开一条纯转发流，\n"
        "把麦克风原声写进主输出（虚拟声卡 Input）——\n"
        "Discord/OBS 等固定录虚拟声卡即可，不用再来回切换输入设备。\n"
        "点「开始」变声时直通流自动让位给推理引擎。"
    )
    g.addWidget(QLabel("停机直通"), r, 0)
    g.addWidget(win.passthrough_check, r, 1, 1, 2)
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
