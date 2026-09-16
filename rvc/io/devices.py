"""音频设备枚举 — 轻量模块，仅依赖 sounddevice，避免启动时加载 torch。"""
import sounddevice as sd


def _rescan_portaudio_devices():
    """重建 PortAudio 设备列表（热插拔后 query_devices 可能返回空）。

    ⚠ 依赖 sounddevice 私有 API `_terminate`/`_initialize`——公开 API 没有
    等价的"重扫"入口。sounddevice 升级前必须核对这两个名字仍存在；
    若上游移除，本函数是唯一的故障点（显式集中，便于排查）。
    """
    sd._terminate()
    sd._initialize()


def get_audio_devices(hostapi_name=None):
    """枚举音频设备，仅在设备列表为空时重新初始化 sounddevice。

    hostapi_name 为 hostapi 显示名（'Windows WASAPI' / 'MME' / ...）；
    传入无效名时回退到第一个可用 hostapi。

    Returns:
        (hostapi 名列表, 输入设备名列表, 输出设备名列表,
         输入设备索引列表, 输出设备索引列表)
    """
    if not sd.query_devices():
        _rescan_portaudio_devices()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for ha in hostapis:
        for idx in ha["devices"]:
            devices[idx]["hostapi_name"] = ha["name"]

    def _has_channels(device, channel_key):
        return device[channel_key] > 0 and device.get("hostapi_name") == filter_name

    ha_names = [h["name"] for h in hostapis]
    filter_name = hostapi_name if hostapi_name in ha_names else ha_names[0]
    inputs = [d["name"] for d in devices if _has_channels(d, "max_input_channels")]
    outputs = [d["name"] for d in devices if _has_channels(d, "max_output_channels")]
    input_indices = [d["index"] for d in devices if _has_channels(d, "max_input_channels")]
    output_indices = [d["index"] for d in devices if _has_channels(d, "max_output_channels")]
    return ha_names, inputs, outputs, input_indices, output_indices
