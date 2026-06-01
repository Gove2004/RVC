"""音频工具 — 设备枚举、相位声码器、声学预设"""
import numpy as np
import sounddevice as sd
import torch

PRESETS = {
    "原声纯净": {"eq_low": 0, "eq_mid": 0, "eq_high": 0, "warmth": 0, "compress": 0, "reverb": 0},
    "温暖电台": {"eq_low": 3, "eq_mid": 1.5, "eq_high": -1, "warmth": 0.35, "compress": 0.5, "reverb": 0.02},
    "贴耳ASMR": {"eq_low": 1, "eq_mid": -1, "eq_high": 4, "warmth": 0.1, "compress": 0.3, "reverb": 0.04},
    "明亮通透": {"eq_low": -2, "eq_mid": 2, "eq_high": 3.5, "warmth": 0.2, "compress": 0.25, "reverb": 0.1},
    "空旷大厅": {"eq_low": -1, "eq_mid": 0, "eq_high": 1.5, "warmth": 0, "compress": 0.15, "reverb": 0.35},
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
