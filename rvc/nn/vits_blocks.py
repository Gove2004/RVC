"""VITS 通用数值块（原版 RVC 拷贝层）。

本文件是原版 RVC 的数值路径拷贝：函数行为与上游逐行等价，重写时禁止
"顺手优化"——slice_segments 等切片实现已锁定现状（见各函数注释），
改动会影响与已导出模型/预训练权重的数值一致性。
"""
from typing import Optional

import torch


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4):
    # AL-4：刻意不做向量化（gather/index 有数值与边界风险且非性能瓶颈），锁死逐样本切片。
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = int(ids_str[i].item())
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = int(ids_str[i].item())
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 4):
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str
