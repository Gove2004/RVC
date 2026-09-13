"""实时 pitch 跟踪与缓存。"""
from rvc.audio.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
import torch

from rvc.inference.f0_extractor import create_f0_extractor

# 实时 pitch 缓存长度：必须 ≥ 最大可能 p_len（16k 缓冲总帧数）。
# p_len = 总时长(extra_time + block + 交叉淡化 + SOLA搜索) × 100 帧/秒；
# UI 上限：extra_time 5.0s + block_time 1.0s + crossfade 0.15s + SOLA搜索 0.01s
# → p_len_max ≈ 616，取 2048 留 3 倍余量。
PITCH_CACHE_SIZE = 2048


def create_pitch_cache(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """创建 F0 缓存（离散 pitch、连续 pitchf、confidence）。"""
    return (
        torch.zeros(PITCH_CACHE_SIZE, device=device, dtype=torch.long),
        torch.zeros(PITCH_CACHE_SIZE, device=device, dtype=torch.float32),
        torch.zeros(PITCH_CACHE_SIZE, device=device, dtype=torch.float32),
    )


def realtime_f0_window(block_frame_16k: int, method: str) -> int:
    """计算实时 F0 提取窗口长度。"""
    frames = block_frame_16k + 800
    if method == "rmvpe":
        frames = 5120 * ((frames - 1) // 5120 + 1) - HUBERT_FRAME_SIZE
    return frames


def update_realtime_pitch_cache_raw(
    input_wav: torch.Tensor,
    block_frame_16k: int,
    p_len: int,
    method: str,
    cache_pitch: torch.Tensor,
    cache_pitchf: torch.Tensor,
    cache_confidence: torch.Tensor,
    device: str,
    is_half: bool,
    inference_cache,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """阶段4：只做 F0 原始提取 + 缓存更新，不做后处理。

    调用 extractor.extract_raw 获取原始 (f0, confidence)，写入缓存。
    后处理（音域映射/中值滤波/离散化）在阶段5由 postprocess_f0 完成。

    Args:
        input_wav: 16k 滚动缓冲区
        block_frame_16k: 本块新增的 16k 样本数
        p_len: 需要的 F0 帧数
        method: F0 提取方法（rmvpe/fcpe）
        cache_pitch/cache_pitchf/cache_confidence: F0 缓存
            - cache_pitchf 写入原始连续 F0
            - cache_confidence 写入原始 confidence
            - cache_pitch 只左移，新值由阶段5后处理后写入
        device/is_half/inference_cache: 设备和缓存
        config: InferenceConfig

    Returns:
        (f0_raw, confidence_raw): 当前块需要的原始 F0 (1, p_len) 和 confidence (1, p_len)
    """
    f0_extractor_frame = realtime_f0_window(block_frame_16k, method)
    extractor = create_f0_extractor(method, device, is_half, inference_cache, config=config)
    f0_raw, confidence_raw = extractor.extract_raw(
        input_wav[-f0_extractor_frame:], HUBERT_SAMPLE_RATE,
    )

    shift = block_frame_16k // HUBERT_FRAME_SIZE
    # 三个缓存同步左移
    cache_pitch[:-shift] = cache_pitch[shift:].clone()
    cache_pitchf[:-shift] = cache_pitchf[shift:].clone()
    cache_confidence[:-shift] = cache_confidence[shift:].clone()

    # 帧对齐：pitch[3:-1] 去掉首尾边缘帧，按偏移 4 写入缓存尾部
    write_start = 4 - f0_raw.shape[0]
    cache_pitchf[write_start:] = f0_raw[3:-1]
    cache_confidence[write_start:] = confidence_raw[3:-1]
    # cache_pitch 只左移，新值由阶段5 postprocess_f0 后写入

    return (
        cache_pitchf[None, -p_len:].clone(),
        cache_confidence[None, -p_len:].clone(),
    )
