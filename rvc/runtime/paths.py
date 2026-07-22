"""运行时路径。"""
from pathlib import Path

CONFIG_ROOT = Path("assets/configs")
STATE_FILE = CONFIG_ROOT / "save_state.json"
TRAIN_CONFIG_FILE = CONFIG_ROOT / "48ktrain_config.json"


def config_path() -> Path:
    return STATE_FILE


def train_config_path(sr: int | str = None) -> Path:
    if not TRAIN_CONFIG_FILE.exists():
        raise FileNotFoundError(f"找不到训练配置: {TRAIN_CONFIG_FILE}")
    return TRAIN_CONFIG_FILE
