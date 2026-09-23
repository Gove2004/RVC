"""音色调节 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子"""
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QPushButton, QComboBox,
)
from PySide6.QtWidgets import QFileDialog

from gui.infer.view.widgets import _create_slider_row, RangeSlider
from gui.styles.components import ButtonStyles
from pathlib import Path

if TYPE_CHECKING:
    from gui.infer.view.contracts import InferWindowHost


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
    """音色调节 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子。"""
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
    win.model_path = ""
    win.model_path_btn = QPushButton("选择模型...")
    win.model_path_btn.setMinimumHeight(28)
    win.model_path_btn.setToolTip("点击选择模型文件 (.pth)")
    win.model_path_btn.setStyleSheet(ButtonStyles.path_button())
    win.model_path_btn.clicked.connect(lambda: _browse_model(win))

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
        20.0, 1000.0, 10.0,
        params.voice.pitch_map_src_min, params.voice.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("原声音域"), r, 0)
    g.addWidget(win.pitch_map_src_range, r, 1)
    g.addWidget(pitch_map_src_label, r, 2)
    r += 1

    # ── 模型音域（原目标音域）──
    win.pitch_map_dst_range, pitch_map_dst_label = _range_row(
        20.0, 1000.0, 10.0,
        params.voice.pitch_map_dst_min, params.voice.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("模型音域"), r, 0)
    g.addWidget(win.pitch_map_dst_range, r, 1)
    g.addWidget(pitch_map_dst_label, r, 2)
    r += 1

    # ── 性别因子（formant）──
    win.formant_slider, formant_label = _create_slider_row(-2.5, 2.5, 0.05, 0.0, fmt="+.2f")
    g.addWidget(QLabel("性别因子"), r, 0)
    g.addWidget(win.formant_slider, r, 1)
    g.addWidget(formant_label, r, 2)
    r += 1

    # ── 响度因子（rms_mix）──
    win.rms_mix_slider, rms_mix_label = _create_slider_row(0.0, 1.0, 0.05, 0.0)
    g.addWidget(QLabel("响度因子"), r, 0)
    g.addWidget(win.rms_mix_slider, r, 1)
    g.addWidget(rms_mix_label, r, 2)
    r += 1

    return w


def _browse_model(win: "InferWindowHost"):
    path, _ = QFileDialog.getOpenFileName(win, "选择模型", str(win.controller.models_dir), "模型 (*.pth)")
    if path:
        win.model_path = path
        win.runtime_params.model_path = path
        win.model_path_btn.setText(Path(path).name)
        win.model_path_btn.setToolTip(path)
