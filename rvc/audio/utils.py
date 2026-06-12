"""音频工具 — 设备枚举、相位声码器、声学预设"""
import numpy as np
import sounddevice as sd
import torch

PRESETS = {
    "原声": {"eq_sub": 0, "eq_low": 0, "eq_mid": 0, "eq_hi_mid": 0, "eq_high": 0},
    "明亮": {"eq_sub": -2, "eq_low": -1, "eq_mid": 0, "eq_hi_mid": 3, "eq_high": 4},
    "温暖": {"eq_sub": 2, "eq_low": 3, "eq_mid": 1, "eq_hi_mid": -1, "eq_high": -2},
    "清脆": {"eq_sub": -3, "eq_low": -1, "eq_mid": 2, "eq_hi_mid": 3, "eq_high": 2},
    "浑厚": {"eq_sub": 3, "eq_low": 2, "eq_mid": 0, "eq_hi_mid": -2, "eq_high": -1},
    "人声增强": {"eq_sub": -2, "eq_low": 0, "eq_mid": 3, "eq_hi_mid": 2, "eq_high": 0},
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
