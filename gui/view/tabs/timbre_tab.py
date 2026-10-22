"""音色 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子 / 输出音量"""
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox

from gui.view.widgets import create_slider_row, RangeSlider, PathPickerButton

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost


def _range_row(min_val, max_val, step, low_val, high_val,
               fmt=".0f", unit="Hz", label_w=80):
    """创建「双滑块范围 + 自动格式化值标签」，返回 (range_slider, label)。"""
    rs = RangeSlider(min_val, max_val, step, low_val, high_val, fmt=fmt, unit=unit)
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignCenter)

    def _fmt(low, high):
        return f"{low:{fmt}}-{high:{fmt}}{unit}"

    lbl.setText(_fmt(rs.low(), rs.high()))
    rs.rangeChanged.connect(lambda low, high: lbl.setText(_fmt(low, high)))
    return rs, lbl


def build_timbre_tab(win: "InferWindowHost"):
    """音色 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子 / 输出音量。"""
    params = win.runtime_params
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 模型路径 + 特征器 ──
    # 路径由按钮自带（path()），不再另存 win.model_path 平行字段；
    # 参数绑定里 model_path 走 PATH 方式直接读按钮。
    win.model_path_btn = PathPickerButton(
        win, title="选择模型", filter_str="模型 (*.pth)",
        placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )

    win.hubert_combo = QComboBox()
    win.hubert_combo.addItems(["base", "chinese"])
    win.hubert_combo.setMinimumHeight(28)
    win.hubert_combo.setToolTip("此模型训练时用的特征器（base=原版 hubert_base，chinese=腾讯中文 hubert）。训练与推理必须一致。")
    win.hubert_combo.setMaximumWidth(90)

    g.addWidget(QLabel("模型路径"), r, 0)
    g.addWidget(win.model_path_btn, r, 1)
    g.addWidget(win.hubert_combo, r, 2)
    r += 1

    # ── 原声音域 ──
    win.pitch_map_src_range, pitch_map_src_label = _range_row(
        10.0, 1000.0, 10.0,
        params.voice.pitch_map_src_min, params.voice.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("原声音域"), r, 0)
    g.addWidget(win.pitch_map_src_range, r, 1)
    g.addWidget(pitch_map_src_label, r, 2)
    r += 1

    # ── 模型音域 ──
    win.pitch_map_dst_range, pitch_map_dst_label = _range_row(
        10.0, 1000.0, 10.0,
        params.voice.pitch_map_dst_min, params.voice.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("模型音域"), r, 0)
    g.addWidget(win.pitch_map_dst_range, r, 1)
    g.addWidget(pitch_map_dst_label, r, 2)
    r += 1

    # ── 性别因子（formant）──
    win.formant_slider, formant_label = create_slider_row(-2.5, 2.5, 0.05, 0.0, fmt="+.2f")
    # 会话级参数：它决定合成器推理图的输出长度，运行中改动会让音频回调线程
    # 现场重捕获 CUDA Graph（实测 585ms ≈ 108 个块），必然断音。运行档禁用
    # （见 MainWindow._refresh_mode），停止后修改、下次启动生效。
    win.formant_slider.setToolTip("性别因子（共振峰偏移）：改动在下次启动变声时生效")
    g.addWidget(QLabel("性别因子"), r, 0)
    g.addWidget(win.formant_slider, r, 1)
    g.addWidget(formant_label, r, 2)
    r += 1

    # ── 响度因子（rms_mix）──
    win.rms_mix_slider, rms_mix_label = create_slider_row(0.0, 1.0, 0.05, 0.0)
    g.addWidget(QLabel("响度因子"), r, 0)
    g.addWidget(win.rms_mix_slider, r, 1)
    g.addWidget(rms_mix_label, r, 2)
    r += 1

    # ── 输出音量（volume；纯标量增益，1=原始，>1 放大，<1 衰减）──
    # 量程经 controller 下发（单源仍是 rvc.core.config，与热路径钳位同一常量）
    vol_min, vol_max = win.controller.volume_range
    win.volume_slider, volume_label = create_slider_row(
        vol_min, vol_max, 0.1, params.volume, fmt=".2f",
    )
    g.addWidget(QLabel("输出音量"), r, 0)
    g.addWidget(win.volume_slider, r, 1)
    g.addWidget(volume_label, r, 2)
    r += 1

    return w
