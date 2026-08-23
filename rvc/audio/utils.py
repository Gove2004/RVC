"""音频工具 — 设备枚举、RMS 响度匹配"""
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from rvc.audio.device_query import get_audio_devices  # noqa: F401 轻量模块，供 rvc.audio 惰性导出


def match_rms(source_audio, source_sr, target_audio, target_sr, mix_rate):
    """RMS 响度匹配 — 统一的实现供实时/离线推理复用。

    Args:
        source_audio: np.ndarray, 原始音频（参考响度）
        source_sr: int, 原始音频采样率
        target_audio: np.ndarray, 转换后音频（需调整响度）
        target_sr: int, 转换后音频采样率
        mix_rate: float, 混合比例 [0, 1]
            0.0 = 完全使用源响度
            1.0 = 完全使用转换后响度

    Returns:
        np.ndarray: 响度调整后的音频
    """
    # 计算 RMS
    rms1 = librosa.feature.rms(y=source_audio, frame_length=source_sr // 2 * 2, hop_length=source_sr // 2)
    rms2 = librosa.feature.rms(y=target_audio, frame_length=target_sr // 2 * 2, hop_length=target_sr // 2)

    # 插值到目标长度
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()

    # 防止除零
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    # 混合响度
    ratio = torch.pow(rms1, 1 - mix_rate) * torch.pow(rms2, mix_rate - 1)
    target_audio = target_audio * ratio.numpy()

    return target_audio
