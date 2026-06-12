"""模型 Tab — 可滚动的 ModelCard 列表"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFileDialog,
)
import os


def build_models_tab(win):
    """构建「模型」Tab，返回 QWidget。控件属性设置到 win 上。"""
    w = QWidget(); l = QVBoxLayout(w); l.setSpacing(6); l.setContentsMargins(8,8,8,8)
    bar = QHBoxLayout()
    bar.addWidget(QLabel("模型列表")); bar.addStretch()
    btn_add = QPushButton("+ 添加模型")
    btn_add.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:3px 8px;border-radius:3px}QPushButton:hover{background:#005f9e}")
    btn_add.clicked.connect(win._add_model)
    bar.addWidget(btn_add)
    l.addLayout(bar)
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
    container = QWidget()
    win._models_layout = QVBoxLayout(container); win._models_layout.addStretch()
    scroll.setWidget(container)
    l.addWidget(scroll, 1)
    win._model_cards = []
    return w
