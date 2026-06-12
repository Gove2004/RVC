"""训练步骤 Tab — 按钮 + 进度 + 日志"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QTextEdit,
)


def build_train_tab(win) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)

    # 按钮行
    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(6)

    win.btn_preprocess = QPushButton("1. 预处理")
    win.btn_preprocess.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:3px 5px;border-radius:3px}QPushButton:hover{background:#005f9e}QPushButton:disabled{background:#555;color:#888}")
    win.btn_f0 = QPushButton("2. 提取F0")
    win.btn_f0.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:3px 5px;border-radius:3px}QPushButton:hover{background:#005f9e}QPushButton:disabled{background:#555;color:#888}")
    win.btn_feature = QPushButton("3. 提取特征")
    win.btn_feature.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:3px 5px;border-radius:3px}QPushButton:hover{background:#005f9e}QPushButton:disabled{background:#555;color:#888}")
    win.btn_train = QPushButton("4. 训练")
    win.btn_train.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:3px 5px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555;color:#888}")
    win.btn_all = QPushButton("一键全流程")
    win.btn_all.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:3px 5px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555;color:#888}")
    win.stop_btn = QPushButton("停止训练")
    win.stop_btn.setStyleSheet("QPushButton{background:#555;color:#888;font-weight:bold;padding:3px 5px;border-radius:3px}")

    win.btn_preprocess.clicked.connect(lambda: win._start_step("preprocess"))
    win.btn_f0.clicked.connect(lambda: win._start_step("f0"))
    win.btn_feature.clicked.connect(lambda: win._start_step("feature"))
    win.btn_train.clicked.connect(lambda: win._start_step("train"))
    win.btn_all.clicked.connect(lambda: win._start_step("all"))
    win.stop_btn.clicked.connect(win.stop_training)
    win.stop_btn.setEnabled(False)

    btn_layout.addWidget(win.btn_preprocess)
    btn_layout.addWidget(win.btn_f0)
    btn_layout.addWidget(win.btn_feature)
    btn_layout.addWidget(win.btn_train)
    btn_layout.addWidget(win.btn_all)
    btn_layout.addWidget(win.stop_btn)
    layout.addLayout(btn_layout)

    # 进度
    layout.addWidget(_build_progress_group(win))

    # 日志
    win.log_edit = QTextEdit()
    win.log_edit.setReadOnly(True)
    log_group = QGroupBox("日志")
    log_layout = QVBoxLayout(log_group)
    log_layout.addWidget(win.log_edit)
    layout.addWidget(log_group, 1)

    return widget


def _build_progress_group(win) -> QGroupBox:
    group = QGroupBox("训练进度")
    layout = QVBoxLayout(group)
    win.stage_label = QLabel("当前阶段: 未开始")
    win.epoch_label = QLabel("Epoch: -")
    win.loss_label = QLabel("Loss: -")
    win.progress_bar = QProgressBar()
    win.progress_bar.setRange(0, 100)
    layout.addWidget(win.stage_label)
    layout.addWidget(win.epoch_label)
    layout.addWidget(win.loss_label)
    layout.addWidget(win.progress_bar)
    return group
