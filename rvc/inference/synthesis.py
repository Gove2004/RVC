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


def infer_synth_audio(
    synthesizer,
    feats,
    p_len_t,
    pitch,
    pitchf,
    sid,
    use_f0: int,
    is_half: bool,
    skip_head: int | None = None,
    return_length: int | None = None,
    return_length2: int | None = None,
):
    """Synthesizer 推理调用（实时/离线共用），走 CUDA Graph（如果启用）。

    离线路径不传 skip_head/return_length/return_length2（synthesizer.infer 走 5 参签名）；
    实时路径三者必传（8 参签名，含 skip_head/return_length/return_length2）。
    """
    realtime = skip_head is not None
    if use_f0 == 1:
        pitch, pitchf = cast_pitch_tensors(pitch, pitchf, is_half)
        if realtime:
            graph_key = "synth-realtime-f0"
            call = lambda: synthesizer.infer(feats, p_len_t, pitch, pitchf, sid, skip_head, return_length, return_length2)
        else:
            graph_key = "synth-offline-f0"
            call = lambda: synthesizer.infer(feats, p_len_t, pitch, pitchf, sid)
    else:
        if realtime:
            graph_key = "synth-realtime-no-f0"
            call = lambda: synthesizer.infer(feats, p_len_t, None, None, sid, skip_head, return_length, return_length2)
        else:
            graph_key = "synth-offline-no-f0"
            call = lambda: synthesizer.infer(feats, p_len_t, None, None, sid)
    return run_cuda_graph(synthesizer, graph_key, call)


def apply_formant_resample(audio: torch.Tensor, factor: float, target_sr: int, resample_kernel: dict, device: str) -> torch.Tensor:
    from torchaudio.transforms import Resample as TatResample

    upp_res = int((factor * target_sr // 100))
    if upp_res == target_sr // 100:
        return audio

    if upp_res not in resample_kernel:
        if len(resample_kernel) >= 64:
            resample_kernel.clear()
        resample_kernel[upp_res] = TatResample(
            orig_freq=upp_res,
            new_freq=target_sr // 100,
            dtype=torch.float32,
        ).to(device)
    return resample_kernel[upp_res](audio)
