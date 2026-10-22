"""F0 工具函数 — 音高提取/归一化的共享逻辑。

推理侧（GPU/torch）与训练侧（CPU/numpy）共用同一套常量与算法，
避免两处独立实现导致的口径漂移。

双后端策略：数学表达式只写一份，`clamp/where/nan_to_num/取整` 等
后端差异经 _Ops 注入（见 _TENSOR_OPS / _ARRAY_OPS）。不要在模块内部
按 `if xp is torch` 把算法复制两份——两边的更新迟早不同步，而本模块
存在的理由就是消灭这种漂移。
"""
from types import ModuleType
from typing import Any

import numpy as np
import torch

from rvc.dsp.hz_midi import hz_to_midi, midi_to_hz
from rvc.models.rmvpe.constants import F0_MEL_MAX, F0_MEL_MIN

# Mel 频率转换常数（HTK 公式：mel = 1127 * ln(1 + f/700)）
MEL_SLOPE = 1127.0
MEL_F0 = 700.0

# 离散 pitch 范围：[1, 255]，0 保留为静音/UV
PITCH_MIN = 1
PITCH_MAX = 255
PITCH_BINS = PITCH_MAX - PITCH_MIN + 1  # 255

# 音域映射缓存上限（MIDI 常量缓存与 (参数,设备) tensor 缓存共用同一上限）
_PITCH_MAP_CACHE_LIMIT = 128

# 音域映射预计算缓存：参数组合 → (src_min_m, src_max_m, dst_min_m, dst_max_m)
# 避免每次调用重复计算 4 次 hz_to_midi（参数在运行时很少变化）
# 无锁设计：GIL 保证单次 dict 读写原子；并发 check-then-set 最坏情况是两个线程
# 各创建一份相同结果，后者覆盖前者，无正确性问题，仅浪费一次计算。
_pitch_map_midi_cache: dict[tuple, tuple] = {}

# 参数组合+设备 → 4 个常量 tensor。torch.tensor(scalar, device=cuda) 是
# 同步 H2D 拷贝，实时路径每块调用 4 次纯属浪费——参数与设备在会话内不变。
# 同上无锁设计。
_pitch_map_tensor_cache: dict[tuple, tuple] = {}


class _Ops:
    """同一套逐元素算子的两种实现（torch / numpy）。

    只收数学上必须分流的算子：能两边同名的（log/log2/乘加）不包一层，
    免得把「后端注入」变成新的样板代码。
    """

    __slots__ = ("xp", "log", "clip", "where", "nan_to_num", "zeros_like",
                 "round_to_int")

    def __init__(self, xp: ModuleType, *, clip, where, nan_to_num, zeros_like,
                 round_to_int):
        self.xp = xp
        self.log = xp.log
        self.clip = clip
        self.where = where
        self.nan_to_num = nan_to_num
        self.zeros_like = zeros_like
        self.round_to_int = round_to_int


_TENSOR_OPS = _Ops(
    torch,
    # hi=None 表示只钳下界（apply_pitch_map 需要「下方兜底、上方不限」）
    clip=lambda t, lo, hi: torch.clamp(t, min=lo, max=hi),
    where=lambda cond, a, b: torch.where(cond, a, b),
    nan_to_num=lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0),
    zeros_like=torch.zeros_like,
    # round 与 numpy rint 同为「四舍六入五取偶」，再取 long
    round_to_int=lambda t: torch.round(t).long(),
)

_ARRAY_OPS = _Ops(
    np,
    clip=lambda t, lo, hi: np.clip(t, lo, hi),
    where=lambda cond, a, b: np.where(cond, a, b),
    nan_to_num=lambda t: np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0),
    zeros_like=lambda t: np.zeros_like(t),
    round_to_int=lambda t: np.rint(t).astype(np.int64),
)


def _ops_for(value: Any) -> _Ops:
    """按输入类型选后端（torch.Tensor → GPU 路径，其余按 numpy 数组处理）。"""
    return _TENSOR_OPS if torch.is_tensor(value) else _ARRAY_OPS


def _evict_oldest(cache: dict) -> None:
    """超上限丢最早插入的项；pop 带 default，容忍并发插入造成的字典变化。"""
    if len(cache) > _PITCH_MAP_CACHE_LIMIT:
        cache.pop(next(iter(cache)), None)


def _pitch_map_midi(key: tuple) -> tuple:
    """参数组合 → 4 个 MIDI 半音值（float，进程内缓存）。"""
    cached = _pitch_map_midi_cache.get(key)
    if cached is not None:
        return cached
    src_min, src_max, dst_min, dst_max = key
    value = (
        float(hz_to_midi(src_min)), float(hz_to_midi(src_max)),
        float(hz_to_midi(dst_min)), float(hz_to_midi(dst_max)),
    )
    _pitch_map_midi_cache[key] = value
    _evict_oldest(_pitch_map_midi_cache)
    return value


def _pitch_map_consts(key: tuple, ops: _Ops, f0: Any) -> tuple:
    """4 个 MIDI 常量：numpy 直接用 float，torch 用同设备 0 维张量。

    张量版本按 (参数, 设备) 缓存，避免每块 4 次同步 H2D 拷贝。
    """
    if ops.xp is np:
        return _pitch_map_midi(key)
    tensor_key = (*key, f0.device)
    cached = _pitch_map_tensor_cache.get(tensor_key)
    if cached is None:
        cached = tuple(
            torch.tensor(v, dtype=torch.float32, device=f0.device)
            for v in _pitch_map_midi(key)
        )
        _pitch_map_tensor_cache[tensor_key] = cached
        _evict_oldest(_pitch_map_tensor_cache)
    return cached


def apply_pitch_map(f0, src_min, src_max, dst_min, dst_max):
    """半音尺度线性音域映射（音程按恒定斜率 k 等比缩放，旋律形状不变）。

    在 MIDI 半音尺度上做线性映射，而不是 Hz 尺度。
    Hz 尺度线性映射 y=kx+b（b≠0）对音程的压缩比随音高变化，旋律变形。
    半音尺度映射 y_semi = k*x_semi + b 把所有音程统一缩放为 k 倍：
    音程**绝对大小不保持**（k<1 时压缩音高动态），但缩放比处处恒定，
    旋律的相对形状（音程之间的比例）不变。k=1（两区间等宽）时才是
    严格意义的移调。

    模型音域侧钳制（不静音）：映射后的输出 F0 超出模型音域
    [dst_min, dst_max] 的帧钳制到最近边界，避免破音的同时保持有声。
    原声音域 [src_min, src_max] 仅用于校准映射比例，不做安全边界。

    ⚠️ out_m = dst_min + ratio * (dst_max - dst_min) 无斜率约束，
    k = 两区间半音宽度之比。用户当前存档 60–720 → 160–720（k=0.6053）
    是他要的效果（已明确裁定），不要以修 bug 名义改动该行为。

    安全保护：
    - 参数下限保护（最小 1Hz）：避免 hz_to_midi(0) = -inf 导致 NaN
    - 参数有效性检查：src_min < src_max 且 dst_min < dst_max，否则原样返回
    - 输出 NaN/inf 过滤：异常帧设为 0（清音），避免 CUDA device-side assert

    Args:
        f0: 输入 F0（torch.Tensor 或 numpy.ndarray，单位 Hz），0=清音/UV
        src_min/src_max: 原声音域（Hz）
        dst_min/dst_max: 目标音域（Hz）

    Returns:
        映射后的 F0，类型与输入一致，清音保持 0，范围外钳制到边界
    """
    # 参数下限保护：避免 0 或负数导致 hz_to_midi 产生 -inf/NaN
    _MIN_FREQ = 1.0
    src_min = max(float(src_min), _MIN_FREQ)
    src_max = max(float(src_max), _MIN_FREQ)
    dst_min = max(float(dst_min), _MIN_FREQ)
    dst_max = max(float(dst_max), _MIN_FREQ)

    # 参数有效性检查：音域下限必须小于上限
    if src_min >= src_max or dst_min >= dst_max:
        return f0  # 无效参数，原样返回

    key = (src_min, src_max, dst_min, dst_max)
    midi = _pitch_map_midi(key)

    # 守卫用 Python float：拿 0 维 CUDA 张量做 if 判断等于每块强同步一次
    if midi[1] - midi[0] < 1e-6:
        return f0  # 原声音域无效，原样返回

    ops = _ops_for(f0)
    c_src_min, c_src_max, c_dst_min, c_dst_max = _pitch_map_consts(key, ops, f0)

    # 清音掩码在钳制前取：钳制把 0 抬到 1e-6，之后就找不到哪些是 UV 帧了
    uv_mask = f0 <= 0
    f0_m = hz_to_midi(ops.clip(f0, 1e-6, None), ops.xp)   # Hz → MIDI 半音

    ratio = (f0_m - c_src_min) / (c_src_max - c_src_min)   # 半音尺度线性映射
    out_m = ops.clip(c_dst_min + ratio * (c_dst_max - c_dst_min), c_dst_min, c_dst_max)

    out = midi_to_hz(out_m, ops.xp)   # MIDI → Hz
    # 原始 UV 帧 + NaN/inf 帧 → 输出 0
    return ops.where(uv_mask, ops.zeros_like(out), ops.nan_to_num(out))


def _f0_to_mel(f0, xp):
    """将 F0(Hz) 转换为 Mel 频率（xp = numpy 或 torch）。"""
    return MEL_SLOPE * xp.log(1 + f0 / MEL_F0)


def median_filter_f0(f0, kernel=3):
    """因果一维中值滤波，去除 F0 孤立误判帧，减少气声和抖动。

    因果实现：只看当前和过去帧，不引入未来延迟。
    清音帧（f0=0）始终保持 0；浊音帧（f0>0）如果被滤波成 0
    （窗口中零值占多数，如浊音帧紧邻清音边界），恢复原值避免误清。

    Args:
        f0: 连续 F0（1D torch.Tensor，Hz），0=清音/UV
        kernel: 中值滤波窗口大小（奇数，默认3）

    Returns:
        滤波后的 F0（形状/设备/类型不变）
    """
    # 非 1-D 输入原样返回（调用链保证 1-D；2-D 会走错 unfold 维度、4-D 直接
    # NotImplementedError，与其让晦涩异常炸实时回调，不如按既有兜底约定透传）
    if not torch.is_tensor(f0) or f0.dim() != 1 or f0.numel() < kernel or kernel < 3:
        return f0
    # 因果 padding：只在左边补 kernel-1 个首帧值（不看未来）
    f0_padded = torch.nn.functional.pad(
        f0.unsqueeze(0).unsqueeze(0), (kernel - 1, 0), mode="replicate"
    ).squeeze()
    windows = f0_padded.unfold(0, kernel, 1)
    median_vals = windows.median(dim=-1).values
    # 清音保持0；浊音被滤波成0时恢复原值
    return torch.where(f0 > 0, torch.where(median_vals > 0, median_vals, f0), f0)


def normalize_f0_to_coarse(f0):
    """将连续 F0(Hz) 归一化为离散 pitch 值 [1, 255]。

    自动适配 torch.Tensor（GPU）和 numpy.ndarray（CPU）输入。
    静音/UV（f0 <= 0）映射为 1。
    NaN/inf 输入视为清音，映射为 1，避免 CUDA device-side assert。

    Args:
        f0: 连续 F0 值（torch.Tensor 或 numpy.ndarray，单位 Hz）

    Returns:
        离散化的 pitch 值，范围 [1, 255]，类型与输入一致
    """
    ops = _ops_for(f0)
    f0_mel = _f0_to_mel(ops.nan_to_num(f0), ops.xp)
    scaled = (f0_mel - F0_MEL_MIN) * (PITCH_BINS - 1) / (F0_MEL_MAX - F0_MEL_MIN) + PITCH_MIN
    # 刻意不用布尔掩码索引（f0_mel[voiced] = ...）：布尔索引要先 nonzero()
    # 定输出形状，是数据依赖形状操作 —— 每块强制同步 GPU 队列，且无法被
    # CUDA Graph 捕获（cudaErrorStreamCaptureInvalidated）。where 全程形状
    # 静态，可进图、无同步。
    pitch = ops.where(f0_mel > 0, scaled, ops.zeros_like(scaled))
    return ops.round_to_int(ops.clip(pitch, PITCH_MIN, PITCH_MAX))
