"""运行时参数容器 — GUI 线程写、音频回调线程读（依赖简单标量更新）"""
from dataclasses import dataclass


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
    break_ratio: float = 0.4             # 高音区压缩比（越小压越狠；1.0 = 不压缩）
    break_knee: float = 0.12             # 平滑膝宽（相对临界比例：膝范围 = 临界×(1±knee)）

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
