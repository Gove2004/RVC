"""音频设备枚举 — 轻量模块，仅依赖 sounddevice，避免启动时加载 torch/librosa。"""
import sounddevice as sd

from rvc.audio.wasapi import HOSTAPI_EXCLUSIVE, is_exclusive_name


def get_audio_devices(hostapi_name=None):
    """枚举音频设备，仅在首次调用时重新初始化 sounddevice。

    hostapi_name 可以是真实 hostapi 名，也可以是「WASAPI（独占模式）」
    合成项（内部映射回真实 WASAPI 过滤设备，独占标志由引擎消费）。
    """
    if not sd.query_devices():
        sd._terminate(); sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for ha in hostapis:
        for idx in ha["devices"]:
            devices[idx]["hostapi_name"] = ha["name"]
    ha_names = [h["name"] for h in hostapis]
    # 注入独占合成项：紧跟在真实 WASAPI 后面（设备过滤仍按真实 WASAPI）
    for i, h in enumerate(ha_names):
        if "wasapi" in h.lower():
            ha_names.insert(i + 1, HOSTAPI_EXCLUSIVE)
            break
    # 独占合成项 → 真实 WASAPI 名（设备过滤依据）
    filter_name = hostapi_name
    if is_exclusive_name(hostapi_name):
        filter_name = next((h for h in ha_names if "wasapi" in h.lower()), hostapi_name)
    if filter_name not in [h for h in ha_names if not is_exclusive_name(h)]:
        filter_name = next((h for h in ha_names if not is_exclusive_name(h)), "")
    filt = lambda d, ch: d[ch] > 0 and d.get("hostapi_name") == filter_name
    inputs = [d["name"] for d in devices if filt(d, "max_input_channels")]
    outputs = [d["name"] for d in devices if filt(d, "max_output_channels")]
    in_idx = [d["index"] for d in devices if filt(d, "max_input_channels")]
    out_idx = [d["index"] for d in devices if filt(d, "max_output_channels")]
    return ha_names, inputs, outputs, in_idx, out_idx
