"""离线 Tab — 离线音频文件转换"""
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QProgressBar, QVBoxLayout,
)

from gui.styles import ButtonStyles, Layout, sep

def build_offline_tab(win):
    w = QWidget()
    root = QVBoxLayout(w); root.setSpacing(4); root.setContentsMargins(8, 8, 8, 8)

    inp_row = QHBoxLayout()
    win.offline_input = QLineEdit()
    btn_in = QPushButton("…"); btn_in.setFixedWidth(36); btn_in.setStyleSheet(ButtonStyles.small())
    btn_in.clicked.connect(lambda: win._off_browse(win.offline_input, "in"))
    inp_row.addWidget(QLabel("输入文件")); inp_row.addWidget(win.offline_input, 1); inp_row.addWidget(btn_in)
    iw = QWidget(); iw.setLayout(inp_row); root.addWidget(iw)

    root.addWidget(sep())

    out_row = QHBoxLayout()
    win.offline_output = QLineEdit()
    btn_out = QPushButton("…"); btn_out.setFixedWidth(36); btn_out.setStyleSheet(ButtonStyles.small())
    btn_out.clicked.connect(lambda: win._off_browse(win.offline_output, "out"))
    out_row.addWidget(QLabel("输出文件")); out_row.addWidget(win.offline_output, 1); out_row.addWidget(btn_out)
    ow = QWidget(); ow.setLayout(out_row); root.addWidget(ow)

    root.addWidget(sep())

    row = QHBoxLayout()
    win.offline_button = QPushButton("开始转换")
    win.offline_button.setFixedWidth(80)
    win.offline_button.setStyleSheet(ButtonStyles.secondary())
    win.offline_button.clicked.connect(win._off_start)
    row.addWidget(win.offline_button)
    win.offline_status = QLabel("")
    row.addWidget(win.offline_status); row.addStretch()
    rw = QWidget(); rw.setLayout(row); root.addWidget(rw)

    win.offline_progress = QProgressBar()
    root.addWidget(win.offline_progress)

    return w
