"""音频设备枚举 — 轻量模块，仅依赖 sounddevice，避免启动时加载 torch/librosa。"""
import sounddevice as sd


def get_audio_devices(hostapi_name=None):
    """枚举音频设备，仅在首次调用时重新初始化 sounddevice。"""
    if not sd.query_devices():
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
