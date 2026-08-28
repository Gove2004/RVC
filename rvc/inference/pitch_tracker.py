"""实时 pitch 跟踪与缓存。"""
import torch

from rvc.inference.f0_extractor import create_f0_extractor

# 实时 pitch 缓存长度：必须 ≥ 最大可能 p_len（16k 缓冲总帧数）。
# p_len = 总时长(extra_time+block+交叉淡化+SOLA搜索) × 100 帧/秒；
# UI 允许 extra_time 最大 15s + block 1.0s → p_len_max ≈ 1616，取 2048 留余量。
# 之前 1024 会在 extra_time 调大时截断 pitchf，导致 protect_blend 维度不匹配（1031 vs 1024）。
PITCH_CACHE_SIZE = 2048


def create_pitch_cache(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(PITCH_CACHE_SIZE, device=device, dtype=torch.long),
        torch.zeros(PITCH_CACHE_SIZE, device=device, dtype=torch.float32),
    )


def extract_f0(x, f0_up_key: float, method: str, device: str, is_half: bool, inference_cache, f0_proc: tuple | None = None):
    extractor = create_f0_extractor(method, device, is_half, inference_cache)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    return extractor.extract(x, 16000, f0_up_key, f0_proc)


def prepare_offline_pitch(input_wav, p_len: int, f0_up_key: float, method: str, device: str, is_half: bool, inference_cache, f0_proc: tuple | None = None):
    pitch, pitchf = extract_f0(input_wav, f0_up_key, method, device, is_half, inference_cache, f0_proc)
    return pitch[:p_len].unsqueeze(0).contiguous(), pitchf[:p_len].unsqueeze(0).contiguous()


def realtime_f0_window(block_frame_16k: int, method: str) -> int:
    frames = block_frame_16k + 800
    if method == "rmvpe":
        frames = 5120 * ((frames - 1) // 5120 + 1) - 160
    return frames


def update_realtime_pitch_cache(
    input_wav: torch.Tensor,
    block_frame_16k: int,
    p_len: int,
    return_length: int,
    return_length2_val: int,
    f0_up_key: float,
    method: str,
    cache_pitch: torch.Tensor,
    cache_pitchf: torch.Tensor,
    device: str,
    is_half: bool,
    inference_cache,
    f0_proc: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    f0_extractor_frame = realtime_f0_window(block_frame_16k, method)
    pitch, pitchf = extract_f0(
        input_wav[-f0_extractor_frame:], f0_up_key, method, device, is_half, inference_cache,
        f0_proc,
    )
    shift = block_frame_16k // 160
    cache_pitch[:-shift] = cache_pitch[shift:].clone()
    cache_pitchf[:-shift] = cache_pitchf[shift:].clone()
    cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
    cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]
    return cache_pitch[None, -p_len:], cache_pitchf[None, -p_len:] * return_length2_val / return_length
