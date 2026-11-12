"""Synthesizer 推理调用。"""
import math

import torch
from torchaudio.transforms import Resample as TatResample

from rvc.runtime.graph import run_cuda_graph

# formant 重采样核缓存上限（达到该长度时整表 clear，见 apply_formant_resample）
_RESAMPLE_KERNEL_LIMIT = 64


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
    use_f0: bool,
    is_half: bool,
    skip_head: int,
    return_length: int,
    return_length2: int,
    z_noise=None,
) -> torch.Tensor:
    """Synthesizer 推理调用，走 CUDA Graph（如果启用）。返回合成音频。

    skip_head/return_length/return_length2 是标量参数，不是张量，
    不能作为 CUDA Graph 输入，必须通过闭包捕获。
    但 feats/p_len_t/pitch/pitchf/sid/z_noise 是张量，必须作为参数传入，
    否则 CUDA Graph 捕获/回放时用的是外部变量地址而非静态张量地址，
    重开时外部变量地址变化导致 CUDA Graph 使用错误数据（声音沙哑/失真）。
    """
    tail = (skip_head, return_length, return_length2)

    # return_length2 必须加入 graph_key：它通过闭包捕获编译进 CUDA Graph，
    # 不同 formant 因子对应不同 return_length2，若共用 key 会错误命中旧图导致输出长度错误、爆音卡顿。
    # skip_head/return_length 同理入 key——虽然它们当前的变化必然连带改变
    # feats 形状（经形状签名触发重捕获），但那是隐式不变量；显式入 key 后
    # 未来引入「不改变缓冲长度的 head 调整」也不会静默错误复用旧图。
    rl2_suffix = f"-rl2-{return_length2}-head-{skip_head}-len-{return_length}"
    if z_noise is not None:
        # 固定噪声场进图：作为图输入张量，回放复用同一份值（确定性渲染）
        rl2_suffix += "-dz"
    if use_f0:
        pitch, pitchf = cast_pitch_tensors(pitch, pitchf, is_half)
        graph_key = f"synth-f0{rl2_suffix}"
        if z_noise is not None:
            tensor_inputs = (feats, p_len_t, pitch, pitchf, sid, z_noise)
            def call(feats_, p_len_t_, pitch_, pitchf_, sid_, z_noise_):
                return synthesizer.infer(
                    feats_, p_len_t_, pitch_, pitchf_, sid_, *tail, z_noise=z_noise_)[0]
        else:
            tensor_inputs = (feats, p_len_t, pitch, pitchf, sid)
            def call(feats_, p_len_t_, pitch_, pitchf_, sid_):
                return synthesizer.infer(feats_, p_len_t_, pitch_, pitchf_, sid_, *tail)[0]
    else:
        graph_key = f"synth-no-f0{rl2_suffix}"
        if z_noise is not None:
            tensor_inputs = (feats, p_len_t, sid, z_noise)
            def call(feats_, p_len_t_, sid_, z_noise_):
                return synthesizer.infer(
                    feats_, p_len_t_, None, None, sid_, *tail, z_noise=z_noise_)[0]
        else:
            tensor_inputs = (feats, p_len_t, sid)
            def call(feats_, p_len_t_, sid_):
                return synthesizer.infer(feats_, p_len_t_, None, None, sid_, *tail)[0]
    # 只回传音频：infer 的 (x_mask, (z, z_p, m_p, logs_p)) 全部是块内中间量。
    # 图回放必须 clone 输出（静态缓冲下一块就被覆写），若把整个 tuple 交出去，
    # 每块会多拷 4 个 (1,192,T) 张量（约 1MB）——拷完立刻变垃圾。
    # 必须把 tensor inputs 传到 run_cuda_graph 里——否则 cache 不认识 tensor shape，
    # synthesize 实际从未进 graph，每次都走 eager，Python 调度开销白白浪费。
    return run_cuda_graph(synthesizer, graph_key, call, *tensor_inputs)


def formant_frames(formant_factor: float, return_length: int) -> tuple[int, float]:
    """formant 因子 → (合成器输出帧数, 量化后的有效共振峰比例)。

    合成器按 `ceil(return_length * factor)` 帧合成、再重采样回 `return_length`
    帧：帧率变了、时长没变，共振峰随比例移动而音高不变（时长拉伸由合成器
    内部对谐波源的 interpolate 抵消，pitchf 不参与）。取整使 formant 的有效
    分辨率 = 1/return_length 帧——一个整数帧数对应一张 CUDA Graph。

    两个返回值必须来自同一次算术：若帧数与重采样比例各算一遍（一处 ceil、
    一处用原始 factor），输出长度会与重采样核错位。
    """
    frames = max(1, int(math.ceil(return_length * formant_factor)))
    return frames, frames / return_length


def formant_resample_ratio(factor: float, sr: int) -> int:
    """formant 重采样的「上采样分母」：orig_freq，单位为 100Hz。

    抽成单源函数：runner 需要它算截断长度、apply_formant_resample 需要它做
    缓存键与构造核，两处若各写一遍算术，日后单边修改就会无声漂移。
    """
    return int(math.floor(factor * sr // 100))


def apply_formant_resample(audio: torch.Tensor, factor: float, sr: int, resample_kernel: dict, device: str) -> torch.Tensor:
    "formant 偏移用重采样实现（保留共振峰位置，改变音高）。"
    sr100 = sr // 100
    upp_res = formant_resample_ratio(factor, sr)
    if upp_res == sr100:
        return audio

    # 缓存键必须含两侧频率：upp_res 由 factor*sr 派生，不同 sr 在
    # 特定 factor 下会撞出同一个 upp_res（如 48000*1.0208 与 44100*1.1111 都得 490），
    # 而核的 new_freq 是 sr//100 —— 只按 upp_res 缓存会复用错误采样率的核，
    # 表现为换采样率模式后 formant 量持续偏差。
    key = (upp_res, sr100)
    if key not in resample_kernel:
        if len(resample_kernel) >= _RESAMPLE_KERNEL_LIMIT:
            resample_kernel.clear()
        resample_kernel[key] = TatResample(
            orig_freq=upp_res,
            new_freq=sr100,
            dtype=torch.float32,
        ).to(device)
    return resample_kernel[key](audio)
