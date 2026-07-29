"""实时 SOLA 对齐与交叉淡化。

SOLA (Short-time Overlap-Add) 是一种语音处理中的时域对齐算法，
通过寻找最佳重叠位置并在过渡区进行交叉淡入淡出，实现语音段平滑拼接。

算法流程：
1. 在搜索窗口内计算参考向量（sola_buffer）与当前推理向量的互相关
2. 找到相关系数最大的偏移位置作为最佳匹配点
3. 如果匹配有效（能量足够且相关性达标），则从该偏移处开始输出
4. 对于重叠部分，若启用相位声码器则进行时频域合成，否则直接线性混合
5. 更新sola_buffer为最新输出块的前部，用于下一帧对齐

关键阈值：
- SOLA_MIN_CORR (0.1): 最小相关系数，低于此值表示无可靠匹配
- SOLA_MIN_ENERGY (1e-4): 最小信号能量，避免在静音区域对齐
"""
import torch
import torch.nn.functional as F

from rvc.audio.utils import phase_vocoder


SOLA_MIN_CORR = 0.1
SOLA_MIN_ENERGY = 1e-4


def apply_sola(
    infer: torch.Tensor,
    sola_buffer: torch.Tensor,
    sola_norm_kernel: torch.Tensor,
    fade_in: torch.Tensor,
    fade_out: torch.Tensor,
    block_samples: int,
    sola_buffer_samples: int,
    sola_search_samples: int,
    use_phase_vocoder: bool,
) -> torch.Tensor:
    """执行单次 SOLA 对齐与交叉淡化。

    Args:
        infer: 当前推理音频块，形状 [block_samples + search_offset]
        sola_buffer: 上一块的末尾缓存，长度 = sola_buffer_samples
        sola_norm_kernel: 归一化核，全1向量，长度 = sola_buffer_samples
        fade_in: 淡入窗函数 [sola_buffer_samples]
        fade_out: 淡出窗函数 [sola_buffer_samples]
        block_samples: 当前输出块大小
        sola_buffer_samples: SOLA缓冲区大小（通常为跨帧长度）
        sola_search_samples: 搜索窗口大小
        use_phase_vocoder: 是否使用相位声码器进行更平滑的混合

    Returns:
        对齐后的音频块，形状 [block_samples]，同时更新 sola_buffer
    """
    ci = infer[None, None, :sola_buffer_samples + sola_search_samples]
    cn = F.conv1d(ci, sola_buffer[None, None, :])
    energy = F.conv1d(ci**2, sola_norm_kernel)
    cd = (energy + 1e-8).rsqrt()
    score = cn[0, 0] / cd[0, 0]
    best_score, offset = torch.max(score, dim=0)
    valid_match = (torch.max(energy) >= SOLA_MIN_ENERGY) & (best_score >= SOLA_MIN_CORR)
    offset = torch.where(valid_match, offset, torch.zeros_like(offset))
    infer = infer[offset:]

    if use_phase_vocoder:
        infer[:sola_buffer_samples] = phase_vocoder(sola_buffer, infer[:sola_buffer_samples], fade_out, fade_in)
    else:
        infer[:sola_buffer_samples] *= fade_in
        infer[:sola_buffer_samples] += sola_buffer * fade_out

    sola_buffer[:] = infer[block_samples:block_samples + sola_buffer_samples]
    return infer[:block_samples]
