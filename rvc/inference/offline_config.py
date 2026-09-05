"""离线推理任务配置。

继承 rvc.inference.params.Params——音效/音高/破音保护等字段只定义一次，
消除实时（Params）与离线（OfflineConfig）两份数据层人工同步的坑（曾漏过 break_*）。
本类仅在 Params 之上追加「任务」字段：输入/输出路径、模型路径、HuBERT 特征器。
"""
from dataclasses import dataclass

from rvc.inference.params import Params


@dataclass
class OfflineConfig(Params):
    """离线推理任务配置 = 音频效果参数（继承 Params）+ 路径/模型信息。"""
    input_path: str = ""
    output_path: str = ""
    model_path: str = ""
    hubert: str = "base"  # HuBERT 特征器: base / chinese（必须与训练时一致）
