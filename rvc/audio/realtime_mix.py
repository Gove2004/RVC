"""实时音量包络混合。

通过RMS（均方根）包络提取，将转换音频的音量动态调整到参考音频的音量水平，
实现自然的音色过渡和音量一致性。rms_mix参数控制混合比例：
- 1.0: 完全跟随参考音量
- 0.0: 保持原始转换音量
- 中间值: 混合效果
"""
import torch
import torch.nn.functional as F


def fast_rms(wav: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """快速计算音频的RMS包络。

    通过在平方后进行平均池化实现，注意避免除以零（加clamping）。

    Args:
        wav: 输入音频张量 [samples]
        frame_length: RMS计算的帧长
        hop_length: 帧移

    Returns:
        RMS包络张量 [frame_count]
    """
    padding = frame_length // 2
    squared = F.pad(wav.unsqueeze(0).unsqueeze(0), (padding, padding), mode="reflect") ** 2
    return torch.sqrt(torch.clamp(F.avg_pool1d(squared, frame_length, hop_length).squeeze(), min=1e-8))


def apply_rms_mix(reference: torch.Tensor, converted: torch.Tensor, rms_mix: float, hz_per_centisecond: int) -> torch.Tensor:
    """应用RMS音量包络混合。

    计算参考音频和转换音频各自的RMS包络，通过线性插值对齐长度后，
    按公式: converted * (ref_rms / conv_rms)^(1 - rms_mix) 进行音量调整。

    Args:
        reference: 参考音频张量 [samples]
        converted: 转换后的音频张量 [samples]
        rms_mix: 混合比例 0~1，1表示完全跟随参考音量
        hz_per_centisecond: 每百分秒的Hz数，用于RMS计算的时间尺度

    Returns:
        音量调整后的转换音频 [samples]
    """
    r1 = fast_rms(reference[:converted.shape[0]], 4 * hz_per_centisecond, hz_per_centisecond)
    r1 = F.interpolate(r1[None, None], size=converted.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = fast_rms(converted, 4 * hz_per_centisecond, hz_per_centisecond)
    r2 = F.interpolate(r2[None, None], size=converted.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = torch.clamp(r2, min=1e-3)
    return converted * torch.pow(r1 / r2, 1 - rms_mix)
