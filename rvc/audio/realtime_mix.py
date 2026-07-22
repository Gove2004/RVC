"""实时音量包络混合。"""
import torch
import torch.nn.functional as F


def fast_rms(wav: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    padding = frame_length // 2
    squared = F.pad(wav.unsqueeze(0).unsqueeze(0), (padding, padding), mode="reflect") ** 2
    return torch.sqrt(torch.clamp(F.avg_pool1d(squared, frame_length, hop_length).squeeze(), min=1e-8))


def apply_rms_mix(reference: torch.Tensor, converted: torch.Tensor, rms_mix: float, zc: int) -> torch.Tensor:
    r1 = fast_rms(reference[:converted.shape[0]], 4 * zc, zc)
    r1 = F.interpolate(r1[None, None], size=converted.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = fast_rms(converted, 4 * zc, zc)
    r2 = F.interpolate(r2[None, None], size=converted.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
    r2 = torch.clamp(r2, min=1e-3)
    return converted * torch.pow(r1 / r2, 1 - rms_mix)
