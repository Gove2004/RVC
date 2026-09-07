"""配置管理"""
from gui.configs.config import load_config, save_config
from gui.configs.train_state import TrainGuiState

__all__ = [
    "TrainGuiState",
    "load_config",
    "save_config",
]
