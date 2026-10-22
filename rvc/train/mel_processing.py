"""训练侧频谱计算 — STFT 线性谱与 Mel 谱。

Hann 窗口和 Mel 滤波器按参数 key 缓存，避免每次重复构造。
"""
import torch

from rvc.dsp.mel import mel_filter_bank

mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, c=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * c)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size):
    """STFT 线性谱（幅度）。

    centering 在此手动完成：pad (n_fft-hop)/2 后以 center=False 调 torch.stft，
    与 center=True 语义一致但避免双重 padding（本函数恒手动 pad，故不提供
    center 参数——传 center=True 会 pad 两次）。
    """
    # STFT 实际以 fp32 执行（下方 fp16 输入会转 float，窗口也恒 .float()），
    # 窗口缓存 key 按实际计算 dtype 取——若按转换前的 y.dtype 做 key，
    # fp16 训练时每个 batch 都会从 fp16 窗口重新 .float() 造一份新张量，
    # 缓存形同虚设。
    key = f"{win_size}_fp32_{y.device}"
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size, dtype=torch.float32, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    if y.dtype == torch.float16:
        y = y.float()
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    key = f"{sampling_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_{spec.dtype}_{spec.device}"
    if key not in mel_basis:
        mel = mel_filter_bank(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    melspec = torch.matmul(mel_basis[key], spec)
    return spectral_normalize_torch(melspec)


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size)
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
