"""人声提纯 Tab — 本地批量提取人声 / 去混响 / 去和声 / 去杂音。

布局沿用离线推理的简洁网格风格：输入输出两行 + 选项 + 开始按钮 + 进度 + 日志。
"""
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QGridLayout, QProgressBar, QTextEdit, QHBoxLayout,
)

from gui.styles import ButtonStyles
from rvc.tools.separate import MODELS, EXTRACT_KEYS, POST_KEYS

_OUT_SR_CHOICES = ("44.1k", "40k", "48k")


def build_vocal_tab(win) -> QWidget:
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(4)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # 输入文件夹
    g.addWidget(QLabel("输入文件夹"), r, 0)
    win.sep_input = QLineEdit()
    win.sep_input.setPlaceholderText("原始素材目录（会递归扫描子目录）")
    g.addWidget(win.sep_input, r, 1)
    btn_in = QPushButton("…")
    btn_in.setFixedWidth(36)
    btn_in.setStyleSheet(ButtonStyles.small())
    btn_in.clicked.connect(lambda: win._sep_browse(win.sep_input))
    g.addWidget(btn_in, r, 2)
    r += 1

    # 输出文件夹
    g.addWidget(QLabel("输出文件夹"), r, 0)
    win.sep_output = QLineEdit()
    win.sep_output.setPlaceholderText("提纯结果输出目录（保持子目录结构）")
    g.addWidget(win.sep_output, r, 1)
    btn_out = QPushButton("…")
    btn_out.setFixedWidth(36)
    btn_out.setStyleSheet(ButtonStyles.small())
    btn_out.clicked.connect(lambda: win._sep_browse(win.sep_output))
    g.addWidget(btn_out, r, 2)
    r += 1

    # 主分离模型
    g.addWidget(QLabel("主分离模型"), r, 0)
    win.sep_model = QComboBox()
    for key in EXTRACT_KEYS:
        spec = MODELS[key]
        win.sep_model.addItem(spec.label, key)
        win.sep_model.setItemData(win.sep_model.count() - 1, spec.note, 3)  # Qt.ToolTipRole
    g.addWidget(win.sep_model, r, 1, 1, 2)
    r += 1

    # 输出采样率
    g.addWidget(QLabel("输出采样率"), r, 0)
    win.sep_out_sr = QComboBox()
    win.sep_out_sr.addItems(_OUT_SR_CHOICES)
    win.sep_out_sr.setToolTip("模型原生 44.1k；选 40k/48k 会额外做一次重采样")
    g.addWidget(win.sep_out_sr, r, 1, 1, 2)
    r += 1

    # 可叠加的后处理勾选项
    g.addWidget(QLabel("附加处理"), r, 0)
    checks = QHBoxLayout()
    checks.setSpacing(10)
    for key in POST_KEYS:
        spec = MODELS[key]
        box = QCheckBox(spec.label.split("（")[0])
        box.setToolTip(f"{spec.note}\n权重文件：assets/separate/{spec.filename}")
        setattr(win, f"sep_do_{key}", box)
        checks.addWidget(box)
    checks.addStretch(1)
    g.addLayout(checks, r, 1, 1, 2)
    r += 1

    # 开始 / 停止
    win.sep_start = QPushButton("开始提纯")
    win.sep_start.setFixedWidth(80)
    win.sep_start.setStyleSheet(ButtonStyles.primary())
    win.sep_start.clicked.connect(win._sep_start)
    g.addWidget(win.sep_start, r, 0)
    win.sep_stop = QPushButton("停止")
    win.sep_stop.setFixedWidth(60)
    win.sep_stop.setStyleSheet(ButtonStyles.muted())
    win.sep_stop.setEnabled(False)
    win.sep_stop.clicked.connect(win._sep_stop)
    g.addWidget(win.sep_stop, r, 1)
    win.sep_status = QLabel("")
    g.addWidget(win.sep_status, r, 2)
    r += 1

    # 进度条
    win.sep_progress = QProgressBar()
    g.addWidget(win.sep_progress, r, 0, 1, 3)
    r += 1

    # 日志
    win.sep_log = QTextEdit()
    win.sep_log.setReadOnly(True)
    win.sep_log.setPlaceholderText("处理日志")
    win.sep_log.setFixedHeight(90)
    g.addWidget(win.sep_log, r, 0, 1, 3)

    return w
