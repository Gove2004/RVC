"""实时音量包络混合。

通过RMS（均方根）包络提取，将转换音频的音量动态调整到参考音频的音量水平，
实现自然的音色过渡和音量一致性。rms_mix参数控制混合比例：
- 0.0: 完全跟随参考音量
- 1.0: 保持原始转换音量
- 中间值: 混合效果
"""
import torch
import torch.nn.functional as F


def fast_rms(wav: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """快速计算音频的RMS包络（支持 1D [N] 和 2D [B, N] 批处理）。

    通过在平方后进行平均池化实现，注意避免除以零（加clamping）。

    Args:
        wav: 输入音频张量 [N] 或 [B, N]
        frame_length: RMS计算的帧长
        hop_length: 帧移

    Returns:
        RMS包络张量 [frame_count] 或 [B, frame_count]
    """
    padding = frame_length // 2
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # reflect padding 要求 pad < 输入长度；过短输入给清晰报错而不是晦涩的 RuntimeError
    if wav.shape[-1] <= padding:
        raise ValueError(
            f"fast_rms: 输入长度 {wav.shape[-1]} 必须 > 帧长一半 {padding}"
            f"（frame_length={frame_length} 的 reflect padding 限制）"
        )
    # pad 结果是本函数私有张量，原地平方省一次 [B,1,N] 分配（数值与 **2 完全一致）
    squared = F.pad(wav.unsqueeze(1), (padding, padding), mode="reflect")
    squared = squared.mul_(squared)
    result = torch.sqrt(torch.clamp(F.avg_pool1d(squared, frame_length, hop_length).squeeze(1), min=1e-8))
    return result.squeeze(0) if squeeze else result


def apply_rms_mix(
    reference: torch.Tensor,
    converted: torch.Tensor,
    rms_mix: float,
    hz_per_centisecond: int,
    ref_hz: int | None = None,
) -> torch.Tensor:
    """应用RMS音量包络混合（实时/离线共用）。

    参考与转换音频的 RMS 合并为一次批处理计算（2 通道 stack），
    比两次独立 fast_rms 少一半 pad/avg_pool kernel launch。

    Args:
        reference: 参考音频张量 [samples]
        converted: 转换后的音频张量 [samples]
        rms_mix: 混合比例 0~1，0表示完全跟随参考音量，1表示保持原始
        hz_per_centisecond: 转换音频的厘秒 Hz（实时路径参考与转换同采样率，直接传 sr//100）
        ref_hz: 参考音频的厘秒 Hz（离线路径源 16k ≠ 目标 sr 时传 160；None = 与转换侧相同）

    Returns:
        音量调整后的转换音频 [samples]
    """
    r_hz = ref_hz if ref_hz is not None else hz_per_centisecond
    n = converted.shape[0]

    # 合并为 [2, N] 批处理：通道0=参考，通道1=转换
    # 参考截断到转换长度，与原逻辑 reference[:converted.shape[0]] 一致
    stacked = torch.stack([reference[:n], converted], dim=0)
    # frame_length 用转换侧的 4*hz（参考侧 r_hz 可能不同，但实时路径 r_hz == hz_per_centisecond）
    # 离线路径 ref_hz != hz 时，两侧帧长不同，不能合并——回退独立计算
    if r_hz == hz_per_centisecond:
        r = fast_rms(stacked, 4 * hz_per_centisecond, hz_per_centisecond)  # [2, M]
        r1 = r[0]
        r2 = r[1]
    else:
        r1 = fast_rms(reference[:n], 4 * r_hz, r_hz)
        r2 = fast_rms(converted, 4 * hz_per_centisecond, hz_per_centisecond)

    # size=N+1 再 [:-1]：linear 插值在 align_corners=True 下首尾锚定到端点，
    # 多取一帧再截断可让第 i 个采样点对应第 i 帧的中心，避免整体偏移半帧。
    r1 = F.interpolate(r1[None, None], size=n + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = F.interpolate(r2[None, None], size=n + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = torch.clamp(r2, min=1e-3)
    return converted * torch.pow(r1 / r2, 1 - rms_mix)
