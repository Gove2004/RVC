"""配置管理器 — 负责加载和保存 GUI 配置"""
from typing import TYPE_CHECKING

from gui.configs import load_config, save_config
from gui.configs.infer_state import InferGuiState

if TYPE_CHECKING:
    from gui.infer.window import MainWindow


class ConfigManager:
    """管理 GUI 配置的加载和保存"""

    def __init__(self, window: 'MainWindow'):
        self.window = window

    def load_config(self) -> None:
        cfg = load_config()
        state = InferGuiState.from_dict(cfg.get("gui", {}))
        self.window.apply_gui_state(state)

    def save_config(self) -> None:
        cfg = load_config()
        cfg["gui"] = self.window.collect_gui_state().to_dict()
        save_config(cfg)
