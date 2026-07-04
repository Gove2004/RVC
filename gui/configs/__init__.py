"""配置管理"""
from gui.configs.config import (
    Config,
    ModelConfig,
    load_config,
    save_config,
    train_config_path,
)

__all__ = [
    "Config",
    "ModelConfig",
    "load_config",
    "save_config",
    "train_config_path",
]
