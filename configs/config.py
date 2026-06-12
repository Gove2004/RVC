import json
import sys
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_CONFIG_ROOT = Path("configs")
_TRAIN_CONFIG_DIR = _CONFIG_ROOT / "train"
_STATE_DIR = _CONFIG_ROOT / "state"
_LEGACY_STATE_FILES = {
    "gui": _CONFIG_ROOT / "inuse" / "gui_config.json",
    "train": _CONFIG_ROOT / "inuse" / "train_config.json",
    "models": _CONFIG_ROOT / "models.json",
}
_STATE_FILES = {
    "gui": "gui.json",
    "train": "train.json",
    "models": "models.json",
}


def train_config_path(sr: int | str) -> Path:
    sr_name = sr if isinstance(sr, str) else ("48k" if sr == 48000 else "32k")
    path = _TRAIN_CONFIG_DIR / f"{sr_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到训练配置: {path}")
    return path


def state_path(name: str) -> Path:
    if name not in _STATE_FILES:
        raise KeyError(f"未知状态文件: {name}")
    path = _STATE_DIR / _STATE_FILES[name]
    if path.exists():
        return path
    legacy = _LEGACY_STATE_FILES.get(name)
    if legacy and legacy.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(legacy, path)
    return path


def load_state_json(name: str, default=None):
    path = state_path(name)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def save_state_json(name: str, data: dict):
    path = state_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_train_config_path(sr: int | str) -> Path:
    return train_config_path(sr)


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = "cuda:0"
        self.is_half = True
        self.gpu_name = None
        self.json_config = self._load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self._device_config()

    @staticmethod
    def _load_config_json() -> dict:
        return {
            "train/48k.json": json.loads(train_config_path("48k").read_text(encoding="utf-8")),
            "train/32k.json": json.loads(train_config_path("32k").read_text(encoding="utf-8")),
        }

    def _device_config(self) -> tuple:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This project requires an NVIDIA GPU.")
            sys.exit(1)

        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        logger.info("GPU: %s", self.gpu_name)

        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory
            / 1024
            / 1024
            / 1024
            + 0.4
        )

        x_pad = 3
        x_query = 10
        x_center = 60
        x_max = 65

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
