"""设备 Tab — 推理显卡 / 音频驱动与设备 / 采样率"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox,
    QPushButton, QSizePolicy,
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
    # 列 stretch 必须显式设置：全 0 时 Qt 把富余宽度分给布局外边距（内容整体居中，
    # 左右边缘各多出几十 px），而不是拉伸控件列——与其他页签观感不一致的根源。
    # 富余宽度全部给控件列（col1），标签列/刷新列只占自身内容宽。
    g.setColumnStretch(0, 0)
    g.setColumnStretch(1, 1)
    g.setColumnStretch(2, 0)
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
    # Fixed：按钮只占文字宽度，不随列拉伸（网格列会为对齐把按钮撑得很宽）
    refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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
    # userData 存语义值（model/device），项文案启动后由 lifecycle 回写成带真实
    # 采样率的文本（「模型 40000」/「设备 48000」）；收集/回填都按 userData。
    win.sr_combo = QComboBox()
    win.sr_combo.addItem("模型", "model")
    win.sr_combo.addItem("设备", "device")
    win.sr_combo.setMinimumHeight(28)
    win.sr_combo.setToolTip("模型：流按模型采样率跑；设备：按输入设备默认采样率跑。")
    g.addWidget(_label("采样率"), r, 0)
    g.addWidget(win.sr_combo, r, 1, 1, 2)
    r += 1

    return w, refresh_btn
