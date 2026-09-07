"""RMVPE 模型常量 — F0 范围等参数

从 rvc.inference.f0_extractor 拆出，供推理侧和训练侧共同使用。
"""
import math

# F0 范围常量
F0_MIN = 50.0  # Hz - 人声最低基频
F0_MAX = 1100.0  # Hz - 人声最高基频

F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)
