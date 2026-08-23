"""音频处理模块 — 加载、流管理、工具函数、效果器基类

注意：子模块（realtime_engine/denoise/sola 等）依赖 torch/librosa，属于重型模块。
本包采用惰性导出，GUI 启动路径上只加载轻量子模块（device_query/sounddevice），
torch 等延迟到真正使用音频处理时再导入。
"""
import importlib

__all__ = [
    "load_audio",
    "load_audio_native",
    "RealtimeEngine",
    "get_audio_devices",
    "match_rms",
    "AudioEffect",
    "SpectralSubtraction",
]

_MODULE_MAP = {
    "load_audio": "rvc.audio.loader",
    "load_audio_native": "rvc.audio.loader",
    "RealtimeEngine": "rvc.audio.realtime_engine",
    "get_audio_devices": "rvc.audio.device_query",
    "match_rms": "rvc.audio.utils",
    "AudioEffect": "rvc.audio.effects",
    "SpectralSubtraction": "rvc.audio.denoise",
}


def __getattr__(name):
    mod_name = _MODULE_MAP.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, name)
