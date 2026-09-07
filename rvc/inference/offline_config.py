"""离线推理任务配置。

直接继承 InferenceConfig——音效/音高/破音保护等字段只定义一次，
消除实时与离线两份数据层人工同步的坑。
本类仅在 InferenceConfig 之上追加「任务」字段：输入/输出路径、模型路径、HuBERT 特征器。
"""
from dataclasses import dataclass

from rvc.config import HUBERT_DEFAULT, InferenceConfig


@dataclass
class OfflineConfig(InferenceConfig):
    """离线推理任务配置 = 音频效果参数（继承 InferenceConfig）+ 路径/模型信息。"""
    input_path: str = ""
    output_path: str = ""
    model_path: str = ""
    hubert: str = HUBERT_DEFAULT
