"""配置管理 - 状态持久化、数据结构、设备配置"""
import json
import logging
import sys
import threading
from pathlib import Path
from typing import TypedDict

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# 数据结构定义
# ============================================================================

class ModelConfig(TypedDict, total=False):
    """模型配置（所有值为 UI 显示值）

    - name: 模型名称
    - pth: 模型文件路径
    - idx: 索引文件路径
    - pitch: 音调偏移（半音，int）
    - index_rate: 检索混合率（0.0-1.0）
    - rms_mix: 响度混合（0.0-1.0）
    - gender: 性别偏移（-2.0 到 +2.0）
    - protect: 辅音保护（0.0-1.0）
    """
    name: str
    pth: str
    idx: str
    pitch: int
    index_rate: float
    rms_mix: float
    gender: float
    protect: float


# ============================================================================
# 文件路径
# ============================================================================

_CONFIG_ROOT = Path("assets/configs")
_STATE_FILE = _CONFIG_ROOT / "save_state.json"
_TRAIN_CONFIG_FILE = _CONFIG_ROOT / "48ktrain_config.json"


def config_path() -> Path:
    """获取状态配置文件路径"""
    return _STATE_FILE


def train_config_path(sr: int | str = None) -> Path:
    """获取训练配置路径（仅支持 48k，参数保留仅为兼容）"""
    if not _TRAIN_CONFIG_FILE.exists():
        raise FileNotFoundError(f"找不到训练配置: {_TRAIN_CONFIG_FILE}")
    return _TRAIN_CONFIG_FILE


# ============================================================================
# 状态持久化
# ============================================================================

def load_config() -> dict:
    """加载配置文件

    返回结构：
    {
        "gui": {...},      # 推理界面配置
        "train": {...},    # 训练界面配置
        "models": [...]    # 模型列表
    }
    """
    if not _STATE_FILE.exists():
        return {"gui": {}, "train": {}, "models": []}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"gui": {}, "train": {}, "models": []}


def save_config(data: dict):
    """保存配置文件"""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# 设备配置（单例）
# ============================================================================

class Config:
    """全局设备配置"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.gpu_name = None
        self.json_config = self._load_train_configs()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self._init_device()

    def _load_train_configs(self) -> dict:
        """加载训练配置"""
        return {
            "48ktrain_config.json": json.loads(train_config_path().read_text(encoding="utf-8")),
        }

    def _init_device(self) -> tuple:
        """初始化 GPU 设备"""
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This project requires an NVIDIA GPU.")
            sys.exit(1)

        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        logger.info("GPU: %s", self.gpu_name)

        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4
        )

        # 根据显存调整参数
        if self.gpu_mem <= 4:
            return 1, 5, 30, 32
        return 3, 10, 60, 65
