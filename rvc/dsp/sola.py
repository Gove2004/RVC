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
听感正是「某几个字带小混响/重影」。

性能：NaN 防护走 GPU 端 `torch.where`。不要写成「打包进一个 bool 向量再
`.tolist()`」——那会在这里引入一次 GPU→CPU 同步。

但 **offset 切片那一处仍然同步**，不要被「0-dim tensor 索引」误导：
`infer[offset:]` 里的 `offset` 是 CUDA 标量，Python 切片要拿到主机端整数，
必然走 `__index__` → `.item()` → 设备同步。实测（GPU 队列里堆 80 次 1500³ 矩阵乘后）：
    x[cuda_0dim_tensor:]   66.5 ms
    int(cuda_0dim_tensor)  70.3 ms     ← 等价
    x[python_int:]          0.017 ms
所以它与显式 `.item()` 完全等价，只是把同步藏进了下标运算。要真正消除这次同步，
就得走 GPU 端**固定长度**对齐（pad + gather/roll 出定长块），代价是每块多一次
GPU 上的搬移。**目前不做**：本函数之后紧接着的 `write_main_output` 走的是同步
D2H（回调线程必须在返回前拿到数据），CPU 在同一位置本来就要等 GPU 排空，
这次同步是被掩盖的、不额外增加端到端延迟。

注意：sola_buffer 被原地改写、返回值与 infer 共享存储（交叉淡化是原地乘法/加法），
调用方不得复用传入的 infer。
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
    """
    ci = infer[None, None, :sola_buffer_samples + sola_search_samples]
    cn = F.conv1d(ci, sola_buffer[None, None, :])
    energy = F.conv1d(ci**2, sola_norm_kernel)
    # 对齐源项目归一化方式：只除以 sqrt(当前块能量)
    cor_den = torch.sqrt(energy + 1e-8)
    score = cn[0, 0] / cor_den[0, 0]
    offset = torch.argmax(score)

    # 这一步会触发设备同步（CUDA 标量转主机整数，见模块 docstring）。保留 0-dim
    # 形式而非显式 `int(offset.item())` 只是为了少写一次转换，两者开销相同。
    infer = infer[offset:]

    head = infer[:sola_buffer_samples]
    tail = infer[block_samples:block_samples + sola_buffer_samples]

    # NaN 防护：GPU 上用 where 分支，替代 tolist() + Python if（零同步）
    buf_nan = torch.isnan(sola_buffer).any()
    head_nan = torch.isnan(head).any()
    tail_nan = torch.isnan(tail).any()
    skip_crossfade = buf_nan | head_nan

    # 安全 buffer：NaN 时用零替代，等价于跳过交叉淡化
    safe_buffer = torch.where(skip_crossfade, torch.zeros_like(sola_buffer), sola_buffer)

    infer[:sola_buffer_samples] *= fade_in
    infer[:sola_buffer_samples] += safe_buffer * fade_out

    # 更新 buffer：NaN 时用零
    safe_tail = torch.where(tail_nan, torch.zeros_like(tail), tail)
    sola_buffer[:] = safe_tail

    return infer[:block_samples]
