"""推理 GUI 状态 — 统一使用 rvc.config.AppConfig。

历史上 InferGuiState 是独立的扁平 dataclass，现已统一到 AppConfig
（嵌套结构：inference / engine / active_model / models）。
本模块仅做 re-export，保持 import 路径兼容。
"""
from rvc.config import AppConfig as InferGuiState

__all__ = ["InferGuiState"]
