"""音频工具 — 设备枚举、相位声码器、声学预设、RMS 响度匹配"""
import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

PRESETS = {
    "原声": {"eq_sub": 0, "eq_low": 0, "eq_mid": 0, "eq_hi_mid": 0, "eq_high": 0},
    "萝莉": {"eq_sub": -8, "eq_low": -5, "eq_mid": 0, "eq_hi_mid": 6, "eq_high": 9},
    "少女": {"eq_sub": -5, "eq_low": -2, "eq_mid": 2, "eq_hi_mid": 5, "eq_high": 7},
    "少御": {"eq_sub": -2, "eq_low": 2, "eq_mid": 3, "eq_hi_mid": 2, "eq_high": 0},
    "御姐": {"eq_sub": 4, "eq_low": 6, "eq_mid": 3, "eq_hi_mid": -1, "eq_high": -4},
    "辣条": {"eq_sub": -6, "eq_low": -1, "eq_mid": 5, "eq_hi_mid": 8, "eq_high": 6},
}


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1, device=a.device) + deltaphase
    t = torch.arange(n, device=a.device).unsqueeze(-1) / n
    return a * (fade_out**2) + b * (fade_in**2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n


def match_rms(source_audio, source_sr, target_audio, target_sr, mix_rate):
    """RMS 响度匹配 — 统一的实现供实时/离线推理复用。

    Args:
        source_audio: np.ndarray, 原始音频（参考响度）
        source_sr: int, 原始音频采样率
        target_audio: np.ndarray, 转换后音频（需调整响度）
        target_sr: int, 转换后音频采样率
        mix_rate: float, 混合比例 [0, 1]
            0.0 = 完全使用源响度
            1.0 = 完全使用转换后响度

    Returns:
        np.ndarray: 响度调整后的音频
    """
    # 计算 RMS
    rms1 = librosa.feature.rms(y=source_audio, frame_length=source_sr // 2 * 2, hop_length=source_sr // 2)
    rms2 = librosa.feature.rms(y=target_audio, frame_length=target_sr // 2 * 2, hop_length=target_sr // 2)

    # 插值到目标长度
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()

    # 防止除零
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    # 混合响度
    ratio = torch.pow(rms1, 1 - mix_rate) * torch.pow(rms2, mix_rate - 1)
    target_audio = target_audio * ratio.numpy()

    return target_audio


def get_audio_devices(hostapi_name=None):
    sd._terminate(); sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for ha in hostapis:
        for idx in ha["devices"]:
            devices[idx]["hostapi_name"] = ha["name"]
    ha_names = [h["name"] for h in hostapis]
    if hostapi_name not in ha_names:
        hostapi_name = ha_names[0] if ha_names else ""
    filt = lambda d, ch: d[ch] > 0 and d.get("hostapi_name") == hostapi_name
    inputs = [d["name"] for d in devices if filt(d, "max_input_channels")]
    outputs = [d["name"] for d in devices if filt(d, "max_output_channels")]
    in_idx = [d["index"] for d in devices if filt(d, "max_input_channels")]
    out_idx = [d["index"] for d in devices if filt(d, "max_output_channels")]
    return ha_names, inputs, outputs, in_idx, out_idx
