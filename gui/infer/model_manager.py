"""模型管理器 — 负责模型卡片的增删改查"""
from typing import List, Optional
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QWidget
import os

from gui.infer.widgets import ModelCard, ModelListData
from rvc.runtime.paths import MODELS_DIR


class ModelManager:
    """管理模型列表和模型卡片的生命周期"""

    def __init__(self, parent: QWidget, models_layout: QVBoxLayout):
        self.parent = parent
        self.models_layout = models_layout
        self.cards: List[ModelCard] = []
        self.active_card: Optional[ModelCard] = None

    def add_model_from_file(self) -> None:
        """从文件选择器添加模型"""
        path, _ = QFileDialog.getOpenFileName(
            self.parent, "选择模型", str(MODELS_DIR), "模型 (*.pth)"
        )
        if not path:
            return
        name = os.path.splitext(os.path.basename(path))[0]
        self.add_card(name=name, pth=path)

    def add_card(
        self,
        name: str = "",
        pth: str = "",
        pitch: float = 12.0,
        gender: float = 0.0,
        hubert: str = "chinese"
    ) -> ModelCard:
        """添加模型卡片到列表"""
        card = ModelCard(
            name, pth, pitch=pitch,
            gender=gender,
            hubert=hubert
        )
        card.load_requested.connect(self._handle_card_load)
        card._del.clicked.connect(lambda: self.remove_card(card))
        self.models_layout.insertWidget(self.models_layout.count() - 1, card)
        self.cards.append(card)
        return card

    def remove_card(self, card: ModelCard) -> None:
        """移除模型卡片"""
        if self.active_card == card:
            self.active_card = None
        self.cards.remove(card)
        self.models_layout.removeWidget(card)
        card.deleteLater()
        self.save_models()

    def _handle_card_load(
        self,
        name: str,
        pth: str,
        pitch: float,
        gender: float,
        hubert: str,
    ) -> None:
        """处理卡片加载请求"""
        if not pth:
            return
        if self.active_card:
            self.active_card.set_active(False)
        for c in self.cards:
            if c.pth_edit.text().strip() == pth:
                c.set_active(True)
                self.active_card = c
                break

    def save_models(self) -> None:
        """保存模型列表到持久化存储"""
        ModelListData.save([c.get_data() for c in self.cards])

    def load_models(self) -> None:
        """从持久化存储加载模型列表"""
        for m in ModelListData.load():
            self.add_card(**m)
