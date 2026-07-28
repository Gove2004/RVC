"""离线 Tab — 离线音频文件转换"""
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QGridLayout, QProgressBar,
)

from gui.styles import ButtonStyles, Layout, sep


def build_offline_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(4)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # 输入文件行
    g.addWidget(QLabel("输入文件"), r, 0)
    win.offline_input = QLineEdit()
    g.addWidget(win.offline_input, r, 1, 1, 2)
    btn_in = QPushButton("…")
    btn_in.setFixedWidth(36)
    btn_in.setStyleSheet(ButtonStyles.small())
    btn_in.clicked.connect(lambda: win._off_browse(win.offline_input, "in"))
    g.addWidget(btn_in, r, 3)
    r += 1

    # 输出文件行
    g.addWidget(QLabel("输出文件"), r, 0)
    win.offline_output = QLineEdit()
    g.addWidget(win.offline_output, r, 1, 1, 2)
    btn_out = QPushButton("…")
    btn_out.setFixedWidth(36)
    btn_out.setStyleSheet(ButtonStyles.small())
    btn_out.clicked.connect(lambda: win._off_browse(win.offline_output, "out"))
    g.addWidget(btn_out, r, 3)
    r += 1

    # 按钮行
    win.offline_button = QPushButton("开始转换")
    win.offline_button.setFixedWidth(80)
    win.offline_button.setStyleSheet(ButtonStyles.secondary())
    win.offline_button.clicked.connect(win._off_start)
    g.addWidget(win.offline_button, r, 0)
    win.offline_status = QLabel("")
    g.addWidget(win.offline_status, r, 1)
    r += 1

    win.offline_progress = QProgressBar()
    g.addWidget(win.offline_progress, r, 0, 1, 4)

    return w