"""离线 Tab — 离线音频文件转换"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QGridLayout, QProgressBar,
)

from gui.styles import ButtonStyles


def build_offline_tab(win):
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # 输入文件行（路径选择按钮，显示文件名）
    win.offline_input_path = ""  # 完整路径存储在这里
    win.offline_input_btn = QPushButton("选择输入文件...")
    win.offline_input_btn.setMinimumHeight(28)
    win.offline_input_btn.setToolTip("点击选择输入音频文件")
    win.offline_input_btn.setStyleSheet(ButtonStyles.path_button())
    win.offline_input_btn.clicked.connect(lambda: win._off_browse(None, "in"))
    g.addWidget(QLabel("输入文件"), r, 0)
    g.addWidget(win.offline_input_btn, r, 1, 1, 3)
    r += 1

    # 输出文件行（输入框 + 浏览按钮，直接用 GridLayout）
    g.addWidget(QLabel("输出文件"), r, 0)
    win.offline_output = QLineEdit()
    win.offline_output.setMinimumHeight(28)
    win.offline_output.setPlaceholderText("输出文件路径（可直接输入）")
    win.offline_output.setFocusPolicy(Qt.StrongFocus)
    btn_out = QPushButton("…")
    btn_out.setFixedWidth(36)
    btn_out.setMinimumHeight(28)
    btn_out.setStyleSheet(ButtonStyles.small())
    btn_out.clicked.connect(lambda: win._off_browse(win.offline_output, "out"))
    g.addWidget(win.offline_output, r, 1, 1, 2)
    g.addWidget(btn_out, r, 3)
    r += 1

    # 开始转换（占满整行）
    win.offline_button = QPushButton("开始转换")
    win.offline_button.setMinimumHeight(32)
    win.offline_button.setStyleSheet(ButtonStyles.primary())
    win.offline_button.clicked.connect(win._off_start)
    g.addWidget(win.offline_button, r, 0, 1, 4)
    r += 1

    # 状态标签
    win.offline_status = QLabel("")
    g.addWidget(win.offline_status, r, 0, 1, 4)
    r += 1

    # 进度条
    win.offline_progress = QProgressBar()
    win.offline_progress.setMinimumHeight(24)
    g.addWidget(win.offline_progress, r, 0, 1, 4)

    return w
