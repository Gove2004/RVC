"""离线推理配置数据类"""
from dataclasses import dataclass


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
    gender: float = 0.0
    break_enable: bool = True   # 破音保护（与实时一致，GUI 源值）
    break_src_hz: float = 300.0
    hubert: str = "base"        # HuBERT 特征器: base / chinese（必须与训练时一致）
