"""模型 Tab — 纯卡片列表，首卡片为 + 添加模型"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal

from gui.styles import ButtonStyles, Layout, Colors, MiscStyles


class AddModelCard(QFrame):
    """添加模型占位卡片"""
    add_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        btn = QPushButton("+ 添加模型")
        btn.setStyleSheet(ButtonStyles.secondary())
        btn.clicked.connect(self.add_requested)
        widget = QWidget()
        v = QVBoxLayout(widget); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(widget)
        self.setFixedHeight(44)
        self.setStyleSheet(MiscStyles.placeholder_frame())


def build_models_tab(win):
    w = QWidget()
    l = QVBoxLayout(w)
    l.setSpacing(4); l.setContentsMargins(Layout.TAB_MARGIN, Layout.TAB_MARGIN, Layout.TAB_MARGIN, Layout.TAB_MARGIN)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    container = QWidget()
    win._models_layout = QVBoxLayout(container)
    win._models_layout.setSpacing(4)

    add_card = AddModelCard()
    add_card.add_requested.connect(win._add_model)
    win._models_layout.insertWidget(0, add_card)
    win._models_layout.addStretch()

    scroll.setWidget(container)
    l.addWidget(scroll)
    return w
