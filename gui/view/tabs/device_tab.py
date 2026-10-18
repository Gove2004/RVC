"""设备 Tab — 推理显卡 / 音频驱动与设备 / 采样率"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox,
    QPushButton, QRadioButton, QButtonGroup, QSizePolicy,
)

from gui.styles import ButtonStyles


def _label(text: str) -> QLabel:
    """行标签 — 宽度钉死在文字宽度（Maximum 不参与拉伸）。

    设备列表是后台异步填充的，窗口先出现后下拉框才就位；若标签列
    允许拉伸，网格重排时文本列会被撑宽。Maximum 让文本列恒为最宽
    标签的文字宽度，多余空间全部让给右侧控件列。
    """
    lbl = QLabel(text)
    lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lbl


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
    g.addWidget(_label("显卡驱动"), r, 0)
    g.addWidget(win.gpu_combo, r, 1, 1, 2)
    r += 1

    # ── 音频驱动（刷新按钮由 window 在 device_catalog 就绪后连信号）──
    win.hostapi_combo = QComboBox()
    win.hostapi_combo.currentTextChanged.connect(win._ha_changed)
    refresh_btn = QPushButton("刷新")
    refresh_btn.setStyleSheet(ButtonStyles.secondary())
    g.addWidget(_label("音频驱动"), r, 0)
    g.addWidget(win.hostapi_combo, r, 1)
    g.addWidget(refresh_btn, r, 2)
    r += 1

    win.input_combo = QComboBox()
    g.addWidget(_label("麦克风"), r, 0)
    g.addWidget(win.input_combo, r, 1, 1, 2)
    r += 1

    win.output_combo = QComboBox()
    g.addWidget(_label("主输出"), r, 0)
    g.addWidget(win.output_combo, r, 1, 1, 2)
    r += 1

    win.output2_combo = QComboBox()
    g.addWidget(_label("副输出"), r, 0)
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

    g.addWidget(_label("采样率"), r, 0)
    g.addWidget(win.sr_model_radio, r, 1)
    g.addWidget(win.sr_device_radio, r, 2)
    r += 1

    return w, refresh_btn
