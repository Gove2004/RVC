"""运行时参数容器 — GUI 线程写、音频回调线程读（依赖简单标量更新）"""
from dataclasses import dataclass


@dataclass
class Params:
    pitch: int = 0
    index_rate: float = 0.0
    rms_mix: float = 0.0
    gender: float = 0.0
    protect: float = 0.5
    f0method: str = "fcpe"
    use_pv: bool = False
    enable_eq: bool = False
    eq_sub: float = 0
    eq_low: float = 0
    eq_mid: float = 0
    eq_hi_mid: float = 0
    eq_high: float = 0
    reverb: float = 0
    bgm_enable: bool = False
    bgm_vol: float = 0.5
    enable_out2: bool = False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


p = Params()
