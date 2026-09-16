"""实时 SOLA 对齐与交叉淡化。

SOLA (Short-time Overlap-Add) 是一种语音处理中的时域对齐算法，
通过寻找最佳重叠位置并在过渡区进行交叉淡入淡出，实现语音段平滑拼接。

算法流程：
1. 在搜索窗口内计算参考向量（sola_buffer）与当前推理向量的互相关
2. 找到相关系数最大的偏移位置作为最佳匹配点
3. 从该偏移处开始输出
4. 重叠部分直接线性混合（交叉淡化）
5. 更新sola_buffer为最新输出块的前部，用于下一帧对齐

注意：不要对 offset 做跨块「粘滞式平滑」（如上块差值过大就沿用上块值）。
模型的块间时间偏移每块都在变（f0 周期相位、模型边界效应，幅度可达整个搜索窗），
SOLA 必须每块独立跟踪；一旦沿用旧值，块边界就会重复或跳缺数毫秒——
听感正是「某几个字带小混响/重影」。详见 .workbuddy/memory 2026-08-29 记录。
"""
import torch
import torch.nn.functional as F



def apply_sola(
    infer: torch.Tensor,
    sola_buffer: torch.Tensor,
    sola_norm_kernel: torch.Tensor,
    fade_in: torch.Tensor,
    fade_out: torch.Tensor,
    block_samples: int,
    sola_buffer_samples: int,
    sola_search_samples: int,
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

    Returns:
        对齐后的音频块，形状 [block_samples]，同时更新 sola_buffer。
        注意：返回值与输入 infer 共享存储且已被原地改写（交叉淡化），
        调用方不得复用传入的 infer；需要保留原值时自行 clone。
        NaN 防护路径返回 clone（不共享存储）。
    """
    ci = infer[None, None, :sola_buffer_samples + sola_search_samples]
    cn = F.conv1d(ci, sola_buffer[None, None, :])
    energy = F.conv1d(ci**2, sola_norm_kernel)
    # 对齐源项目归一化方式：只除以 sqrt(当前块能量)
    cor_den = torch.sqrt(energy + 1e-8)
    score = cn[0, 0] / cor_den[0, 0]
    offset = torch.argmax(score)
    offset = int(offset.item())  # 0-d tensor 转 Python int，切片索引更清晰

    infer = infer[offset:]

    # NaN 防护：sola_buffer 被污染时跳过交叉淡化，避免 NaN 恶性循环
    # （两次独立 isnan 扫描合并为一次：.any() 各触发一次 GPU→CPU 同步）
    if torch.isnan(torch.cat((sola_buffer, infer[:sola_buffer_samples]))).any():
        sola_buffer.zero_()
        return infer[:block_samples].clone()

    infer[:sola_buffer_samples] *= fade_in
    infer[:sola_buffer_samples] += sola_buffer * fade_out

    new_tail = infer[block_samples:block_samples + sola_buffer_samples]
    if torch.isnan(new_tail).any():
        sola_buffer.zero_()
    else:
        sola_buffer[:] = new_tail
    return infer[:block_samples]
