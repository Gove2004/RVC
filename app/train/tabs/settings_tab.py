"""训练设置 Tab — 数据设置 + 训练参数"""
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox,
)

from app.train.widgets import browse_directory, browse_file


def build_settings_tab(win) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    layout.addWidget(_build_data_group(win))
    layout.addWidget(_build_train_group(win))
    layout.addStretch(1)
    return widget


def _build_data_group(win) -> QGroupBox:
    group = QGroupBox("数据设置")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(6)

    win.exp_name = QLineEdit("test")
    win.input_dir = QLineEdit()
    browse = QPushButton("浏览")
    browse.setFixedWidth(50)
    browse.clicked.connect(lambda: browse_directory(win, win.input_dir))
    win.sample_rate = QComboBox()
    win.sample_rate.addItems(["48k", "32k"])
    win.sample_rate.currentTextChanged.connect(win._on_sr_changed)

    grid.addWidget(QLabel("实验名"), 0, 0)
    grid.addWidget(win.exp_name, 0, 1, 1, 2)
    grid.addWidget(QLabel("音频目录"), 1, 0)
    grid.addWidget(win.input_dir, 1, 1)
    grid.addWidget(browse, 1, 2)
    grid.addWidget(QLabel("采样率"), 2, 0)
    grid.addWidget(win.sample_rate, 2, 1, 1, 2)
    return group


def _build_train_group(win) -> QGroupBox:
    group = QGroupBox("训练参数")
    form = QFormLayout(group)
    form.setHorizontalSpacing(6)
    form.setVerticalSpacing(6)

    win.epochs = QSpinBox()
    win.epochs.setRange(1, 100000)
    win.epochs.setValue(20)
    win.batch_size = QSpinBox()
    win.batch_size.setRange(1, 64)
    win.batch_size.setValue(1)
    win.save_every = QSpinBox()
    win.save_every.setRange(1, 100000)
    win.save_every.setValue(2)
    win.learning_rate = QLineEdit("1e-4")

    win.pretrain_g = QLineEdit()
    win.pretrain_d = QLineEdit()
    g_row = QHBoxLayout()
    g_btn = QPushButton("浏览")
    g_btn.setFixedWidth(50)
    g_btn.clicked.connect(lambda: browse_file(win, win.pretrain_g))
    g_row.addWidget(win.pretrain_g)
    g_row.addWidget(g_btn)
    d_row = QHBoxLayout()
    d_btn = QPushButton("浏览")
    d_btn.setFixedWidth(50)
    d_btn.clicked.connect(lambda: browse_file(win, win.pretrain_d))
    d_row.addWidget(win.pretrain_d)
    d_row.addWidget(d_btn)

    form.addRow("Epoch", win.epochs)
    form.addRow("Batch size", win.batch_size)
    form.addRow("保存频率", win.save_every)
    form.addRow("学习率", win.learning_rate)
    form.addRow("预训练 G", g_row)
    form.addRow("预训练 D", d_row)
    return group
