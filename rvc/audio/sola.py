"""实时 SOLA 对齐与交叉淡化。"""
import torch
import torch.nn.functional as F

from rvc.audio.utils import phase_vocoder


SOLA_MIN_CORR = 0.1
SOLA_MIN_ENERGY = 1e-4


def apply_sola(
    infer: torch.Tensor,
    sola_buffer: torch.Tensor,
    sola_norm_kernel: torch.Tensor,
    fade_in: torch.Tensor,
    fade_out: torch.Tensor,
    block_frame: int,
    sola_buffer_frame: int,
    sola_search_frame: int,
    use_phase_vocoder: bool,
) -> torch.Tensor:
    ci = infer[None, None, :sola_buffer_frame + sola_search_frame]
    cn = F.conv1d(ci, sola_buffer[None, None, :])
    energy = F.conv1d(ci**2, sola_norm_kernel)
    cd = torch.sqrt(energy + 1e-8)
    score = cn[0, 0] / cd[0, 0]
    best_score, offset = torch.max(score, dim=0)
    valid_match = (torch.max(energy) >= SOLA_MIN_ENERGY) & (best_score >= SOLA_MIN_CORR)
    offset = torch.where(valid_match, offset, torch.zeros_like(offset))
    infer = infer[offset:]

    if use_phase_vocoder:
        infer[:sola_buffer_frame] = phase_vocoder(sola_buffer, infer[:sola_buffer_frame], fade_out, fade_in)
    else:
        infer[:sola_buffer_frame] *= fade_in
        infer[:sola_buffer_frame] += sola_buffer * fade_out

    sola_buffer[:] = infer[block_frame:block_frame + sola_buffer_frame]
    return infer[:block_frame]
