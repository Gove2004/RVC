"""WASAPI — Windows 独占模式辅助（采样率联动 + 失败降级）。

独占模式经 `sd.WasapiSettings(exclusive=True)` 走系统底层，绕过混音器
/重采样链，输出延迟大幅收敛；但要求设备原生采样率与流一致，否则
PortAudio 抛 InvalidSampleRate。本模块提供：
- settings_builder：按开关产出 extra_settings（含存在性探测）；
- 降级判定在 RealtimeEngine.setup（打开失败后去掉独占重试 shared）。

UI 侧「WASAPI（独占模式）」是 hostapi 下拉里的合成项，映射回真实
WASAPI hostapi 过滤设备，并把 exclusive=True 透传进引擎。
"""

from __future__ import annotations

from typing import Any

# hostapi 下拉里合成项的显示名（原版 hostapi_combo 复用此名即可被持久化）
HOSTAPI_EXCLUSIVE = "WASAPI（独占模式）"


def settings_builder(exclusive: bool) -> Any | None:
    """返回传给 sd.Stream(extra_settings=...) 的对象；未启用/环境不支持 → None。"""
    if not exclusive:
        return None
    try:
        import sounddevice as sd
        return sd.WasapiSettings(exclusive=True)
    except Exception:
        # 环境无 sounddevice 或 API 缺失 → 视为不支持独占，静默降级
        return None


def hostapi_name(device_index: int) -> str:
    """设备所在 hostapi 名（'Windows WASAPI' / 'MME' / ...）。"""
    import sounddevice as sd
    info = sd.query_devices(device_index)
    api = sd.query_hostapis(info["hostapi"])
    return api["name"]


def default_samplerate(device_index: int) -> float:
    """设备默认采样率（PortAudio 报告）。"""
    import sounddevice as sd
    info = sd.query_devices(device_index)
    rate = info.get("default_samplerate", 0.0)
    return float(rate) if rate else 48000.0


def is_exclusive_name(name: str) -> bool:
    """下拉文本是否为独占合成项。"""
    return name == HOSTAPI_EXCLUSIVE


__all__ = ["HOSTAPI_EXCLUSIVE", "settings_builder", "hostapi_name",
           "default_samplerate", "is_exclusive_name"]
