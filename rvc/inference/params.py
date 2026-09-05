"""运行时参数容器 — GUI 线程写、音频回调线程读（依赖简单标量更新）"""
from dataclasses import dataclass


@dataclass
class Params:
    pitch: float = 0.0
    rms_mix: float = 0.0
    gender: float = 0.0
    protect: float = 0.5
    f0method: str = "fcpe"
    nr_enable: bool = False
    nr_strength: float = 0.5
    enable_out2: bool = False
    # 破音保护（核心瑕疵：高音破音/沙哑）——源赫兹临界，超过压平
    break_enable: bool = True
    break_src_hz: float = 300.0

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
