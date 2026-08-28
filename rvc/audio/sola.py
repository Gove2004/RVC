"""实时 SOLA 对齐与交叉淡化。

SOLA (Short-time Overlap-Add) 是一种语音处理中的时域对齐算法，
通过寻找最佳重叠位置并在过渡区进行交叉淡入淡出，实现语音段平滑拼接。

算法流程：
1. 在搜索窗口内计算参考向量（sola_buffer）与当前推理向量的互相关
2. 找到相关系数最大的偏移位置作为最佳匹配点
3. 如果匹配有效（能量足够且相关性达标），则从该偏移处开始输出
4. 重叠部分直接线性混合（交叉淡化）
5. 更新sola_buffer为最新输出块的前部，用于下一帧对齐

关键阈值：
- SOLA_MIN_CORR (0.4): 最小相关系数，低于此值表示无可靠匹配（静音/过渡段不偏移，原样拼接）
- SOLA_MIN_ENERGY (1e-3): 最小信号能量，避免在静音区域对齐
- offset 相邻块平滑：offset 与上一块差 > 搜索窗一半时，沿用上一块 offset，
  避免每块时值抖动（声母 20~50ms，±10ms 扰动会糊字）
"""
import torch
import torch.nn.functional as F


SOLA_MIN_CORR = 0.4
SOLA_MIN_ENERGY = 1e-3


def apply_sola(
    infer: torch.Tensor,
    sola_buffer: torch.Tensor,
    sola_norm_kernel: torch.Tensor,
    fade_in: torch.Tensor,
    fade_out: torch.Tensor,
    block_samples: int,
    sola_buffer_samples: int,
    sola_search_samples: int,
    last_offset: torch.Tensor | None = None,
    corr_threshold: float = SOLA_MIN_CORR,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        last_offset: 上一块的 offset（0 维 GPU tensor），用于相邻块平滑；None 则不平滑
        corr_threshold: 相关性阈值（0~1），低于此值判无效匹配（静音/过渡不偏移）

    Returns:
        (对齐后的音频块 [block_samples], 本块 offset) —— offset 供下一块平滑使用
    """
    ci = infer[None, None, :sola_buffer_samples + sola_search_samples]
    cn = F.conv1d(ci, sola_buffer[None, None, :])
    energy = F.conv1d(ci**2, sola_norm_kernel)
    # 标准归一化互相关：除以 sqrt(当前块能量 × 参考能量)，score ∈ [-1, 1]，
    # 阈值语义才成立（旧实现只除当前块能量且除反，score 量级依赖参考能量，阈值形同虚设）
    ref_energy = (sola_buffer**2).sum()
    cd = (energy * ref_energy + 1e-8).rsqrt()
    score = cn[0, 0] * cd[0, 0]
    best_score, offset = torch.max(score, dim=0)
    valid_match = (torch.max(energy) >= SOLA_MIN_ENERGY) & (best_score >= corr_threshold)
    offset.masked_fill_(~valid_match, 0)  # 原地清零，避免每回调分配

    # offset 相邻块平滑：变化超过搜索窗一半时沿用上一块，声母时值不跳变。
    # 静音/无效匹配块：重置为 0（无延续性，重新开始），避免污染下一语音块的平滑状态。
    # 全 GPU 张量操作，避免 CPU↔GPU 同步。第一块（last_offset=None）不平滑。
    if last_offset is not None:
        big_jump = (offset - last_offset).abs() > sola_search_samples // 2
        reset = ~valid_match
        offset = torch.where(reset, offset.new_zeros(()), torch.where(big_jump, last_offset, offset))

    infer = infer[offset:]

    infer[:sola_buffer_samples] *= fade_in
    infer[:sola_buffer_samples] += sola_buffer * fade_out

    sola_buffer[:] = infer[block_samples:block_samples + sola_buffer_samples]
    return infer[:block_samples], offset
