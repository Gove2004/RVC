"""离线 Tab — 离线音频文件转换"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QProgressBar,
)

from gui.styles import ButtonStyles, Layout


def build_offline_tab(win):
    """构建「离线」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); g = QGridLayout(w); g.setSpacing(Layout.SPACING_NORMAL); g.setContentsMargins(Layout.TAB_MARGIN, Layout.TAB_MARGIN, Layout.TAB_MARGIN, Layout.TAB_MARGIN)
    g.setColumnStretch(1, 1); r = 0

    g.addWidget(QLabel("输入文件"), r, 0)
    win.offline_input = QLineEdit(); g.addWidget(win.offline_input, r, 1)
    b = QPushButton("…"); b.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    b.setStyleSheet(ButtonStyles.small())
    b.clicked.connect(lambda: win._off_browse(win.offline_input, "in"))
    g.addWidget(b, r, 2); r += 1

    g.addWidget(QLabel("输出文件"), r, 0)
    win.offline_output = QLineEdit(); g.addWidget(win.offline_output, r, 1)
    b = QPushButton("…"); b.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    b.setStyleSheet(ButtonStyles.small())
    b.clicked.connect(lambda: win._off_browse(win.offline_output, "out"))
    g.addWidget(b, r, 2); r += 1

    row = QHBoxLayout()
    win.offline_button = QPushButton("开始转换")
    win.offline_button.setFixedWidth(80)
    win.offline_button.setStyleSheet(ButtonStyles.secondary())
    win.offline_button.clicked.connect(win._off_start)
    row.addWidget(win.offline_button)
    win.offline_status = QLabel("")
    row.addWidget(win.offline_status)
    row.addStretch()
    g.addLayout(row, r, 0, 1, 3); r += 1

    win.offline_progress = QProgressBar(); win.offline_progress.setValue(0)
    g.addWidget(win.offline_progress, r, 0, 1, 3)
    return w
