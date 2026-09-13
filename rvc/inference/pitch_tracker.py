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


def extract_f0(x, method: str, device: str, is_half: bool, inference_cache, config=None):
    """提取 F0，返回 (pitch, pitchf, confidence) 三元组。"""
    extractor = create_f0_extractor(method, device, is_half, inference_cache, config=config)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    pitch, pitchf, confidence = extractor.extract(x, HUBERT_SAMPLE_RATE)
    return pitch, pitchf, confidence


def realtime_f0_window(block_frame_16k: int, method: str) -> int:
    """计算实时 F0 提取窗口长度。"""
    frames = block_frame_16k + 800
    if method == "rmvpe":
        frames = 5120 * ((frames - 1) // 5120 + 1) - HUBERT_FRAME_SIZE
    return frames


def update_realtime_pitch_cache(
    input_wav: torch.Tensor,
    block_frame_16k: int,
    p_len: int,
    return_length: int,
    return_length2_val: int,
    method: str,
    cache_pitch: torch.Tensor,
    cache_pitchf: torch.Tensor,
    cache_confidence: torch.Tensor,
    device: str,
    is_half: bool,
    inference_cache,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """更新实时 F0 缓存，返回当前块需要的 (pitch, pitchf, confidence)。

    Args:
        input_wav: 16k 滚动缓冲区
        block_frame_16k: 本块新增的 16k 样本数
        p_len: 需要的 F0 帧数
        return_length: 返回帧数（用于 pitchf 缩放）
        return_length2_val: formant 调整后的返回帧数
        method: F0 提取方法（rmvpe/fcpe）
        cache_pitch/cache_pitchf/cache_confidence: F0 缓存
        device/is_half/inference_cache: 设备和缓存
        config: InferenceConfig

    Returns:
        (cache_pitch[None, -p_len:], cache_pitchf[None, -p_len:] * scale, cache_confidence[None, -p_len:])
    """
    f0_extractor_frame = realtime_f0_window(block_frame_16k, method)
    pitch, pitchf, confidence = extract_f0(
        input_wav[-f0_extractor_frame:], method, device, is_half, inference_cache, config=config,
    )
    shift = block_frame_16k // HUBERT_FRAME_SIZE
    cache_pitch[:-shift] = cache_pitch[shift:].clone()
    cache_pitchf[:-shift] = cache_pitchf[shift:].clone()
    cache_confidence[:-shift] = cache_confidence[shift:].clone()
    # 帧对齐：F0 提取器输出与 HuBERT 特征帧存在固定偏移，
    # pitch[3:-1] 去掉首尾边缘帧，按偏移 4 写入缓存尾部。
    cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
    cache_pitchf[4 - pitchf.shape[0]:] = pitchf[3:-1]
    cache_confidence[4 - confidence.shape[0]:] = confidence[3:-1]
    return (
        cache_pitch[None, -p_len:],
        cache_pitchf[None, -p_len:] * return_length2_val / return_length,
        cache_confidence[None, -p_len:],
    )
