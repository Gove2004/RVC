"""全局参数 Tab — 模型路径/特征器 + 采样/融合参数"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QPushButton, QComboBox,
)
from PySide6.QtWidgets import QFileDialog

from gui.infer.view.widgets import _create_slider_row
from gui.styles.components import ButtonStyles
from pathlib import Path


def build_global_params_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 模型路径 + 特征器（同一行，最上面，路径选择按钮）──
    win.model_path = ""  # 完整路径存储在这里（ATTR 绑定）
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

    # ── 采样与融合参数 ──
    win.block_time_slider = _create_slider_row(win, "block_time_slider", 0.05, 0.50, 0.01, 0.25)
    g.addWidget(QLabel("采样长度"), r, 0); g.addWidget(win.block_time_slider, r, 1); g.addWidget(win.block_time_label, r, 2); r += 1

    win.crossfade_slider = _create_slider_row(win, "crossfade_slider", 0.01, 0.05, 0.01, 0.05)
    g.addWidget(QLabel("淡入长度"), r, 0); g.addWidget(win.crossfade_slider, r, 1); g.addWidget(win.crossfade_label, r, 2); r += 1

    win.extra_time_slider = _create_slider_row(win, "extra_time_slider", 0.10, 5.0, 0.10, 2.5)
    g.addWidget(QLabel("额外上下文"), r, 0); g.addWidget(win.extra_time_slider, r, 1); g.addWidget(win.extra_time_label, r, 2); r += 1

    win.rms_mix_slider = _create_slider_row(win, "rms_mix_slider", 0.0, 1.0, 0.05, 0.0)
    g.addWidget(QLabel("响度因子"), r, 0); g.addWidget(win.rms_mix_slider, r, 1); g.addWidget(win.rms_mix_label, r, 2); r += 1

    # 性别因子（formant）：共振峰缩放，-2.5~+2.5，0=不变
    win.formant_slider = _create_slider_row(win, "formant_slider", -2.5, 2.5, 0.05, 0.0, fmt="+.2f")
    g.addWidget(QLabel("性别因子"), r, 0); g.addWidget(win.formant_slider, r, 1); g.addWidget(win.formant_label, r, 2); r += 1

    return w


def _browse_model(win):
    # 浏览起始目录经 controller 取（D5：View 不直接 import rvc 路径常量）
    path, _ = QFileDialog.getOpenFileName(win, "选择模型", str(win.controller.models_dir), "模型 (*.pth)")
    if path:
        win.model_path = path
        # 同步更新 runtime_params（ATTR 类型无控件信号，需手动同步）
        if hasattr(win, 'runtime_params'):
            win.runtime_params.model_path = path
        # 按钮上只显示文件名，完整路径存储在 win.model_path
        win.model_path_btn.setText(Path(path).name)
        win.model_path_btn.setToolTip(path)
