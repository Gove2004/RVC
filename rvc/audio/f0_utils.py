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

# MIDI 参考：A4 = 440Hz = MIDI 69
MIDI_REF_FREQ = 440.0
MIDI_REF_NOTE = 69


def hz_to_midi(freq, xp=None):
    """Hz → MIDI 半音（A4=440Hz = MIDI 69）。自动适配 torch/numpy。"""
    if xp is None:
        xp = torch if torch.is_tensor(freq) else np
    if xp is torch and not torch.is_tensor(freq):
        freq = torch.tensor(freq, dtype=torch.float32)
    return 12.0 * xp.log2(freq / MIDI_REF_FREQ) + MIDI_REF_NOTE


def midi_to_hz(midi, xp=None):
    """MIDI 半音 → Hz。自动适配 torch/numpy。"""
    if xp is None:
        xp = torch if torch.is_tensor(midi) else np
    if xp is torch and not torch.is_tensor(midi):
        midi = torch.tensor(midi, dtype=torch.float32)
    return MIDI_REF_FREQ * (2.0 ** ((midi - MIDI_REF_NOTE) / 12.0))


def apply_pitch_map(f0, src_min, src_max, dst_min, dst_max):
    """半音尺度线性音域映射（保持音程不变，唱歌不跑调）。

    在 MIDI 半音尺度上做线性映射，而不是 Hz 尺度。
    Hz 尺度线性映射 y=kx+b（b≠0）会破坏半音音程，导致跑调。
    半音尺度映射 y_semi = k*x_semi + b 保持音程比例，旋律不变形。

    Args:
        f0: 输入 F0（torch.Tensor 或 numpy.ndarray，单位 Hz），0=清音/UV
        src_min/src_max: 原声音域（Hz）
        dst_min/dst_max: 目标音域（Hz）

    Returns:
        映射后的 F0，类型与输入一致，清音保持 0
    """
    if torch.is_tensor(f0):
        xp = torch
        uv_mask = f0 <= 0
        f0_safe = xp.clamp(f0, min=1e-6)
        # 标量参数转成和 f0 同设备的 tensor，避免 CPU/CUDA 设备不匹配
        device = f0.device
        src_min = torch.tensor(src_min, dtype=torch.float32, device=device)
        src_max = torch.tensor(src_max, dtype=torch.float32, device=device)
        dst_min = torch.tensor(dst_min, dtype=torch.float32, device=device)
        dst_max = torch.tensor(dst_max, dtype=torch.float32, device=device)
    else:
        xp = np
        uv_mask = f0 <= 0
        f0_safe = xp.clip(f0, 1e-6, None)

    # Hz → MIDI 半音
    src_min_m = hz_to_midi(src_min, xp)
    src_max_m = hz_to_midi(src_max, xp)
    dst_min_m = hz_to_midi(dst_min, xp)
    dst_max_m = hz_to_midi(dst_max, xp)
    f0_m = hz_to_midi(f0_safe, xp)

    # 半音尺度线性映射
    src_range = src_max_m - src_min_m
    if src_range < 1e-6:
        return f0  # 原声音域无效，原样返回
    ratio = (f0_m - src_min_m) / src_range
    out_m = dst_min_m + ratio * (dst_max_m - dst_min_m)

    # 两端钳制到目标音域
    out_m = xp.clamp(out_m, dst_min_m, dst_max_m) if xp is torch else xp.clip(out_m, dst_min_m, dst_max_m)

    # MIDI → Hz
    out = midi_to_hz(out_m, xp)
    out[uv_mask] = 0
    return out


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
