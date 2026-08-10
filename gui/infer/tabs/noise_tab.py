"""噪音 Tab — 输入降噪（频谱减）+ 背景音频/底噪"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
)

from gui.infer.widgets import _sl
from gui.styles import ButtonStyles


def build_noise_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(4)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 频谱降噪（输入侧，GPU 谱减法） ──
    win.nr_enable_checkbox = QCheckBox("频谱降噪")
    g.addWidget(win.nr_enable_checkbox, r, 0)
    win.nr_strength_slider = _sl(0, 100, 1, 50)
    win.nr_strength_label = QLabel("0.50")
    win.nr_strength_label.setMinimumWidth(35)
    win.nr_strength_slider.valueChanged.connect(lambda v: win.nr_strength_label.setText(f"{v / 100:.2f}"))
    g.addWidget(win.nr_strength_slider, r, 1, 1, 2)
    g.addWidget(win.nr_strength_label, r, 3)
    r += 1

    # ── 背景音频（文件选择） ──
    g.addWidget(QLabel("背景音频"), r, 0)
    win.bgm_path_edit = QLineEdit()
    win.bgm_path_edit.setPlaceholderText("选择本地音频文件（wav/mp3/flac…）")
    g.addWidget(win.bgm_path_edit, r, 1, 1, 2)
    btn = QPushButton("…")
    btn.setFixedWidth(36)
    btn.setStyleSheet(ButtonStyles.small())
    btn.clicked.connect(win._bgm_browse)
    g.addWidget(btn, r, 3)
    r += 1

    # ── 背景底噪（启用 + 音量，运行中可实时调节） ──
    win.bgm_enable_checkbox = QCheckBox("背景底噪")
    win.bgm_enable_checkbox.stateChanged.connect(
        lambda: win.runtime_params.update(bgm_enable=win.bgm_enable_checkbox.isChecked())
    )
    g.addWidget(win.bgm_enable_checkbox, r, 0)
    win.bgm_vol_slider = _sl(0, 100, 5, 50)
    win.bgm_vol_label = QLabel("0.50")
    win.bgm_vol_label.setMinimumWidth(35)
    win.bgm_vol_slider.valueChanged.connect(lambda v: win.bgm_vol_label.setText(f"{v / 100:.2f}"))
    win.bgm_vol_slider.valueChanged.connect(lambda v: win.runtime_params.update(bgm_vol=v / 100))
    g.addWidget(win.bgm_vol_slider, r, 1, 1, 2)
    g.addWidget(win.bgm_vol_label, r, 3)
    r += 1

    return w
