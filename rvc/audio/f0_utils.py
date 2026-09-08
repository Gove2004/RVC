"""F0 工具函数 — 音高提取/归一化的共享逻辑。

推理侧（GPU/torch）与训练侧（CPU/numpy）共用同一套常量与算法，
避免两处独立实现导致的口径漂移。
"""
import numpy as np
import torch

from rvc.models.rmvpe.constants import F0_MEL_MAX, F0_MEL_MIN

# Mel 频率转换常数（HTK 公式：mel = 1127 * ln(1 + f/700)）
MEL_SLOPE = 1127.0
MEL_F0 = 700.0

# 离散 pitch 范围：[1, 255]，0 保留为静音/UV
PITCH_MIN = 1
PITCH_MAX = 255
PITCH_BINS = PITCH_MAX - PITCH_MIN + 1  # 255

# RMVPE F0 提取阈值：低于此值的置信度判为静音/UV
# 0.03 是 RMVPE 官方推荐值，与 FCPE 的 0.025 同档（底噪全判 uv）
RMVPE_THRESHOLD = 0.03


def _f0_to_mel(f0, xp):
    """将 F0(Hz) 转换为 Mel 频率（xp = numpy 或 torch）。"""
    return MEL_SLOPE * xp.log(1 + f0 / MEL_F0)


def normalize_f0_to_coarse(f0):
    """将连续 F0(Hz) 归一化为离散 pitch 值 [1, 255]。

    自动适配 torch.Tensor（GPU）和 numpy.ndarray（CPU）输入。
    静音/UV（f0 <= 0）映射为 1。

    Args:
        f0: 连续 F0 值（torch.Tensor 或 numpy.ndarray，单位 Hz）

    Returns:
        离散化的 pitch 值，范围 [1, 255]，类型与输入一致
    """
    if torch.is_tensor(f0):
        xp = torch
        f0_mel = _f0_to_mel(f0, xp)
        voiced = f0_mel > 0
        f0_mel[voiced] = (f0_mel[voiced] - F0_MEL_MIN) * (PITCH_BINS - 1) / (F0_MEL_MAX - F0_MEL_MIN) + PITCH_MIN
        f0_mel = xp.clamp(f0_mel, PITCH_MIN, PITCH_MAX)
        return xp.round(f0_mel).long()
    else:
        xp = np
        f0_mel = _f0_to_mel(f0, xp)
        voiced = f0_mel > 0
        f0_mel[voiced] = (f0_mel[voiced] - F0_MEL_MIN) * (PITCH_BINS - 1) / (F0_MEL_MAX - F0_MEL_MIN) + PITCH_MIN
        f0_mel = xp.clip(f0_mel, PITCH_MIN, PITCH_MAX)
        return xp.rint(f0_mel).astype(np.int64)
