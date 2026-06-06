import torch
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, c=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * c)


def dynamic_range_decompression_torch(x, c=1):
    return torch.exp(x) / c


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    key = f"{win_size}_{y.dtype}_{y.device}"
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

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
        window=hann_window[key].float(),
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    key = f"{fmax}_{spec.dtype}_{spec.device}"
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    melspec = torch.matmul(mel_basis[key], spec)
    return spectral_normalize_torch(melspec)


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
