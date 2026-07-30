"""背景 Tab — 本地音频文件作为背景音，循环叠加到最终输出"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
)

from gui.infer.widgets import _sl
from gui.styles import ButtonStyles


def build_bgm_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(4)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # ── 启用开关 ──
    win.bgm_enable_checkbox = QCheckBox("开启背景音")
    win.bgm_enable_checkbox.stateChanged.connect(
        lambda: win.runtime_params.update(bgm_enable=win.bgm_enable_checkbox.isChecked())
    )
    g.addWidget(win.bgm_enable_checkbox, r, 0, 1, 2)
    r += 1

    # ── 音频文件 ──
    g.addWidget(QLabel("音频文件"), r, 0)
    win.bgm_path_edit = QLineEdit()
    win.bgm_path_edit.setPlaceholderText("选择本地音频文件（wav/mp3/flac…）")
    g.addWidget(win.bgm_path_edit, r, 1, 1, 2)
    btn = QPushButton("…")
    btn.setFixedWidth(36)
    btn.setStyleSheet(ButtonStyles.small())
    btn.clicked.connect(win._bgm_browse)
    g.addWidget(btn, r, 3)
    r += 1

    # ── 音量 ──（运行中可实时调节）
    win.bgm_vol_slider = _sl(0, 100, 5, 50)
    win.bgm_vol_label = QLabel("0.50")
    win.bgm_vol_label.setMinimumWidth(35)
    win.bgm_vol_slider.valueChanged.connect(lambda v: win.bgm_vol_label.setText(f"{v / 100:.2f}"))
    win.bgm_vol_slider.valueChanged.connect(lambda v: win.runtime_params.update(bgm_vol=v / 100))
    g.addWidget(QLabel("背景音量"), r, 0)
    g.addWidget(win.bgm_vol_slider, r, 1, 1, 2)
    g.addWidget(win.bgm_vol_label, r, 3)
    r += 1

    hint = QLabel("背景音在启动时载入并循环播放，音量可实时调节。")
    hint.setWordWrap(True)
    g.addWidget(hint, r, 0, 1, 4)

    return w
