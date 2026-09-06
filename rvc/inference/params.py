"""运行时参数容器 — GUI 线程写、音频回调线程读（依赖简单标量更新）"""
from dataclasses import dataclass

# HuBERT 特征器默认档（base=原版 hubert_base / chinese=腾讯中文 hubert）。
# 全项目默认值的唯一来源：pipeline / realtime_engine / offline_config / GUI 均引用此常量，
# 禁止再各自写死 "base"/"chinese" 字面量（曾两套默认并存）。
HUBERT_DEFAULT = "chinese"


@dataclass
class Params:
    pitch: int = 0
    rms_mix: float = 0.0
    gender: float = 0.0
    protect: float = 0.5
    f0method: str = "rmvpe"
    nr_enable: bool = False
    nr_strength: float = 0.5
    enable_out2: bool = False
    # 破音保护（核心瑕疵：高音破音/沙哑）——源赫兹临界，超过后按压缩比软收敛
    break_enable: bool = True
    break_src_hz: float = 300.0          # 破音临界（源 Hz，extractor 内按 key 换算变声后）

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
