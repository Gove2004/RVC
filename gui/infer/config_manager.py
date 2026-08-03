"""配置管理器 — 负责加载和保存 GUI 配置"""
from typing import TYPE_CHECKING

from gui.configs import load_config, save_config
from gui.infer.param_binding import state_from_dict, state_to_dict

if TYPE_CHECKING:
    from gui.infer.window import MainWindow


class ConfigManager:
    """管理 GUI 配置的加载和保存"""

    def __init__(self, window: 'MainWindow'):
        self.window = window

    def load_config(self) -> None:
        cfg = load_config()
        state = state_from_dict(cfg.get("gui", {}))
        self.window.apply_gui_state(state)

    def save_config(self) -> None:
        cfg = load_config()
        cfg["gui"] = state_to_dict(self.window.collect_gui_state())
        save_config(cfg)
