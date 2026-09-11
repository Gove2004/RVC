"""清浊分析 — F0 confidence + 因果中值滤波的 voiced/unvoiced 概率计算。

学术依据：
- Morrison et al. (IEEE TASLP): periodicity 应是连续概率而非二元判定（penn 库）
- Graf et al. (EURASIP JASP 2015): 多特征融合优于单一特征
- Takeda (ATR TR-I-0081): 时间平滑提高清浊边界检测准确率
- Bitra & Beigi: 用 confidence scores 作为可靠性信号抑制 noisy/unvoiced 帧
"""

import torch
import torch.nn.functional as F


def causal_median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """因果中值滤波 — 只看当前和过去帧，不引入未来延迟。

    用于清浊概率的时间平滑，消除孤立误判帧（弹舌音中间偶发浊音帧）。
    中值滤波是非线性滤波：保留真实阶跃边界（不像线性滤波那样模糊边界），
    同时去除脉冲噪声（孤立误判帧），比线性平滑更适合清浊状态。

    Args:
        x: 输入张量，在最后一维做滤波
        kernel_size: 滤波窗口（帧数），3 = 30ms

    Returns:
        滤波后张量，形状不变
    """
    if kernel_size <= 1:
        return x
    original_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    # 因果 padding：左边补 kernel_size-1 个首帧值（不看未来）
    padded = F.pad(x_flat, (kernel_size - 1, 0), mode="replicate")
    windows = padded.unfold(-1, kernel_size, 1)  # (N, T, kernel_size)
    result = windows.median(dim=-1).values
    return result.reshape(original_shape)


def causal_moving_average(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """因果移动平均 — 只看当前和过去帧，用于平滑清浊边界。

    中值滤波保留阶跃边界，但清浊切换处需要渐变来避免咔哒声。
    在中值滤波之后做轻度移动平均，恢复边界处的渐变过渡。

    Args:
        x: 输入张量，在最后一维做平均
        kernel_size: 平均窗口（帧数），3 = 30ms

    Returns:
        平滑后张量，形状不变
    """
    if kernel_size <= 1:
        return x
    original_shape = x.shape
    x_flat = x.reshape(-1, 1, x.shape[-1])  # (N, 1, T) for conv1d
    kernel = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    padded = F.pad(x_flat, (kernel_size - 1, 0), mode="replicate")
    result = F.conv1d(padded, kernel).squeeze(1)
    return result.reshape(original_shape)


def compute_uv_prob(
    pitchf: torch.Tensor,
    confidence: torch.Tensor,
    threshold_hz: float = 20.0,
    width_hz: float = 30.0,
    conf_threshold: float = 0.05,
    median_kernel: int = 3,
) -> torch.Tensor:
    """计算清浊概率 uv_prob（0=浊音，1=清音/UV）。

    融合策略：
    1. F0 基 uv_prob（sigmoid 软阈值逻辑）
    2. F0 confidence 修正 — confidence 低但 F0 非零的帧提高 UV 概率
       中心 = conf_threshold * 2，与 RMVPE 清浊判定阈值联动
    3. 因果中值滤波 — 消除孤立误判帧
    4. 因果移动平均 — 平滑清浊边界，减少咔哒声

    Args:
        pitchf: 连续 F0 (1, T) Hz，0=UV
        confidence: F0 提取器置信度 (1, T) 0~1
        threshold_hz: F0 软阈值中心（Hz）
        width_hz: F0 软阈值宽度（Hz，sigmoid 4σ）
        conf_threshold: RMVPE 清浊判定阈值，confidence 修正中心 = conf_threshold * 2
        median_kernel: 因果中值滤波窗口大小（帧数）

    Returns:
        uv_prob: (1, T) 0~1，0=浊音（全转换），1=清音（按 protect 混合原特征）
    """
    # 特征 1：F0 基 uv_prob（软阈值 sigmoid）
    uv_f0 = torch.sigmoid((threshold_hz - pitchf) / (width_hz / 4))

    # 特征 2：confidence 修正
    # confidence 低但 F0 非零的帧 → 提取器不确定 → 提高 UV 概率保护
    # 中心与 RMVPE 阈值联动：conf_center = conf_threshold * 2
    # 默认 conf_threshold=0.05 → center=0.1, width=0.03，与原硬编码一致
    conf_center = conf_threshold * 2.0
    conf_width = conf_threshold * 0.6
    uv_conf = torch.sigmoid((conf_center - confidence) / conf_width)
    uv_conf = uv_conf * (pitchf > 0).float()  # F0=0 帧 uv_f0 已接近 1
    uv_prob = torch.maximum(uv_f0, uv_conf)

    # 特征 3：因果中值滤波 — 消除孤立误判帧，保留真实清浊边界
    uv_prob = causal_median_filter(uv_prob, median_kernel)

    # 特征 4：因果移动平均 — 平滑清浊边界，减少中值滤波阶跃导致的咔哒声
    uv_prob = causal_moving_average(uv_prob, kernel_size=3)

    return uv_prob
