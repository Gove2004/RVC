"""运行时配置与路径。

device_config 依赖 torch（CUDA 探测），惰性导出；
paths 轻量，启动即加载。
"""
from rvc.runtime.paths import train_config_path

__all__ = ["RuntimeDeviceConfig", "train_config_path"]


def __getattr__(name: str) -> object:
    if name == "RuntimeDeviceConfig":
        from rvc.runtime.device import RuntimeDeviceConfig
        return RuntimeDeviceConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
