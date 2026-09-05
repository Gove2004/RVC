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
    # call 必须接受位置参数 — CUDA Graph fallback (`capture 失败 / 探测失败`)
    # 会用 `function(*inputs)` 调用它，把 tensor inputs 透传过去。
    # 我们用 lambda 接住后忽略，由闭包捕获 feats/p_len_t/... 这些外层 tensor。
    # `tail` 区分实时（多 3 个裁剪参数）与离线（5 参签名），避免重复分支。
    tail = (skip_head, return_length, return_length2) if realtime else ()
    mode = "realtime" if realtime else "offline"

    if use_f0 == 1:
        pitch, pitchf = cast_pitch_tensors(pitch, pitchf, is_half)
        graph_key = f"synth-{mode}-f0"
        tensor_inputs = (feats, p_len_t, pitch, pitchf, sid)
        call = lambda *_: synthesizer.infer(feats, p_len_t, pitch, pitchf, sid, *tail)
    else:
        graph_key = f"synth-{mode}-no-f0"
        tensor_inputs = (feats, p_len_t, sid)
        call = lambda *_: synthesizer.infer(feats, p_len_t, None, None, sid, *tail)
    # 必须把 tensor inputs 传到 run_cuda_graph 里——否则 cache 不认识 tensor shape，
    # synthesize 实际从未进 graph，每次都走 eager，Python 调度开销白白浪费。
    return run_cuda_graph(synthesizer, graph_key, call, *tensor_inputs)


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
