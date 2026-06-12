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
    win.off_in = QLineEdit(); g.addWidget(win.off_in, r, 1)
    b = QPushButton("…"); b.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    b.setStyleSheet(ButtonStyles.small())
    b.clicked.connect(lambda: win._off_browse(win.off_in, "in"))
    g.addWidget(b, r, 2); r += 1

    g.addWidget(QLabel("输出文件"), r, 0)
    win.off_out = QLineEdit(); g.addWidget(win.off_out, r, 1)
    b = QPushButton("…"); b.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    b.setStyleSheet(ButtonStyles.small())
    b.clicked.connect(lambda: win._off_browse(win.off_out, "out"))
    g.addWidget(b, r, 2); r += 1

    row = QHBoxLayout()
    win.off_btn = QPushButton("开始转换")
    win.off_btn.setFixedWidth(80)
    win.off_btn.setStyleSheet(ButtonStyles.secondary())
    win.off_btn.clicked.connect(win._off_start)
    row.addWidget(win.off_btn)
    win.off_status = QLabel("")
    row.addWidget(win.off_status)
    row.addStretch()
    g.addLayout(row, r, 0, 1, 3); r += 1

    win.off_progress = QProgressBar(); win.off_progress.setValue(0)
    g.addWidget(win.off_progress, r, 0, 1, 3)
    return w
