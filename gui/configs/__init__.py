"""配置管理"""
from gui.configs.config import load_config, save_config
from gui.configs.infer_state import InferGuiState
from gui.configs.train_state import TrainGuiState

__all__ = [
    "InferGuiState",
    "TrainGuiState",
    "load_config",
    "save_config",
]
