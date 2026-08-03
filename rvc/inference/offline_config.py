"""离线推理配置数据类"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class OfflineConfig:
    """离线推理配置参数"""
    input_path: str
    output_path: str
    model_path: str
    index_path: str
    pitch: int
    f0method: str
    index_rate: float
    rms_mix: float
    protect: float
    eq_enabled: bool
    eq_bands: Dict[str, float]
    reverb_mix: float
    gender: float = 0.0
