"""Synthesizer 推理调用。"""
import torch

from rvc.tools.cuda_graph import run_cuda_graph


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


def infer_offline_audio(synthesizer, feats, p_len_t, pitch, pitchf, sid, use_f0: int, is_half: bool):
    """离线推理调用，走 CUDA Graph（如果启用）。"""
    if use_f0 == 1:
        pitch, pitchf = cast_pitch_tensors(pitch, pitchf, is_half)
        result = run_cuda_graph(
            synthesizer, "synth-offline-f0",
            lambda: synthesizer.infer(feats, p_len_t, pitch, pitchf, sid),
        )
    else:
        result = run_cuda_graph(
            synthesizer, "synth-offline-no-f0",
            lambda: synthesizer.infer(feats, p_len_t, None, None, sid),
        )
    return result


def infer_realtime_audio(
    synthesizer,
    feats,
    p_len_t,
    cache_pitch,
    cache_pitchf,
    sid,
    skip_head_t,
    return_length_t,
    return_length2,
    use_f0: int,
    is_half: bool,
):
    """实时推理调用，走 CUDA Graph（如果启用）。"""
    if use_f0 == 1:
        cache_pitch, cache_pitchf = cast_pitch_tensors(cache_pitch, cache_pitchf, is_half)
        result = run_cuda_graph(
            synthesizer, "synth-realtime-f0",
            lambda: synthesizer.infer(
                feats, p_len_t, cache_pitch, cache_pitchf, sid,
                skip_head_t, return_length_t, return_length2,
            ),
        )
    else:
        result = run_cuda_graph(
            synthesizer, "synth-realtime-no-f0",
            lambda: synthesizer.infer(
                feats, p_len_t, None, None, sid,
                skip_head_t, return_length_t, return_length2,
            ),
        )
    return result


def apply_formant_resample(audio: torch.Tensor, factor: float, target_sr: int, resample_kernel: dict, device: str) -> torch.Tensor:
    from torchaudio.transforms import Resample as TatResample

    upp_res = int((factor * target_sr // 100))
    if upp_res == target_sr // 100:
        return audio

    if upp_res not in resample_kernel:
        if len(resample_kernel) >= 16:
            resample_kernel.clear()
        resample_kernel[upp_res] = TatResample(
            orig_freq=upp_res,
            new_freq=target_sr // 100,
            dtype=torch.float32,
        ).to(device)
    return resample_kernel[upp_res](audio)
