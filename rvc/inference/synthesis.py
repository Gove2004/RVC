"""Synthesizer 推理调用。"""
import torch
from torchaudio.transforms import Resample as TatResample


def cached_long_tensor(cache: dict, value: int, device: str) -> torch.Tensor:
    value = int(value)
    tensor = cache.get(value)
    if tensor is None:
        tensor = torch.tensor([value], dtype=torch.long, device=device)
        cache[value] = tensor
    return tensor


def cast_pitch_tensors(pitch: torch.Tensor, pitchf: torch.Tensor, is_half: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if is_half:
        return pitch.long(), pitchf.half()
    return pitch.long(), pitchf.float()


def infer_offline_audio(net_g, feats, p_len_t, pitch, pitchf, sid, if_f0: int, is_half: bool):
    if if_f0 == 1:
        pitch, pitchf = cast_pitch_tensors(pitch, pitchf, is_half)
        return net_g.infer(feats, p_len_t, pitch, pitchf, sid)
    return net_g.infer(feats, p_len_t, None, None, sid)


def infer_realtime_audio(
    net_g,
    feats,
    p_len_t,
    cache_pitch,
    cache_pitchf,
    sid,
    skip_head_t,
    return_length_t,
    return_length2,
    if_f0: int,
    is_half: bool,
):
    if if_f0 == 1:
        cache_pitch, cache_pitchf = cast_pitch_tensors(cache_pitch, cache_pitchf, is_half)
        return net_g.infer(
            feats, p_len_t, cache_pitch, cache_pitchf, sid,
            skip_head_t, return_length_t, return_length2,
        )
    return net_g.infer(
        feats, p_len_t, None, None, sid,
        skip_head_t, return_length_t, return_length2,
    )


def apply_formant_resample(audio: torch.Tensor, factor: float, tgt_sr: int, resample_kernel: dict, device: str) -> torch.Tensor:
    upp_res = int((factor * tgt_sr // 100))
    if upp_res == tgt_sr // 100:
        return audio

    if upp_res not in resample_kernel:
        if len(resample_kernel) >= 16:
            resample_kernel.clear()
        resample_kernel[upp_res] = TatResample(
            orig_freq=upp_res,
            new_freq=tgt_sr // 100,
            dtype=torch.float32,
        ).to(device)
    return resample_kernel[upp_res](audio)
