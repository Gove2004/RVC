"""音频处理模块 — 加载、流管理、推理、效果器、降噪、WAV 读写

公开 API（惰性导出，避免 GUI 启动时加载 torch）：
- load_audio: 用 ffmpeg 解码任意格式音频到 numpy 数组
- write_wav / read_wav_info / read_audio_info: WAV 文件读写与元信息
- RealtimeEngine: 实时变声引擎（门面类）
- InferenceRunner: 推理执行器（实时/离线统一）
- AudioStreamManager: 音频流管理（输入/输出/副输出）
- AudioProcessor: 音频效果器链（Denoise/RmsMix/Sola）
- DenoiseEffect / RmsMixEffect / SolaEffect: 单个效果器
- SpectralSubtraction: 谱减法降噪
- get_audio_devices: 查询音频设备列表

其他内部类从子模块直接导入。
"""
import importlib

__all__ = [
    "load_audio",
    "write_wav",
    "read_wav_info",
    "read_audio_info",
    "RealtimeEngine",
    "InferenceRunner",
    "AudioStreamManager",
    "AudioProcessor",
    "DenoiseEffect",
    "RmsMixEffect",
    "SolaEffect",
    "SpectralSubtraction",
    "get_audio_devices",
]

_MODULE_MAP = {
    "load_audio": "rvc.audio.loader",
    "write_wav": "rvc.audio.wav_io",
    "read_wav_info": "rvc.audio.wav_io",
    "read_audio_info": "rvc.audio.wav_io",
    "RealtimeEngine": "rvc.audio.realtime_engine",
    "InferenceRunner": "rvc.audio.inference_runner",
    "AudioStreamManager": "rvc.audio.stream_manager",
    "AudioProcessor": "rvc.audio.effects",
    "DenoiseEffect": "rvc.audio.effects",
    "RmsMixEffect": "rvc.audio.effects",
    "SolaEffect": "rvc.audio.effects",
    "SpectralSubtraction": "rvc.audio.denoise",
    "get_audio_devices": "rvc.audio.device_query",
}


def __getattr__(name):
    mod_name = _MODULE_MAP.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, name)
