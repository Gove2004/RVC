"""统一错误类型体系。

历史上错误处理散落在各处（raise RuntimeError / Exception + 字符串消息），
调用方无法精确区分错误类型。本模块定义 RVC 错误层级，关键路径抛出
具体错误类型，GUI 层可据此给出更精准的用户提示。

错误层级:
    RVCError
    ├── ModelLoadError       模型加载失败（pth 损坏/缺失/不兼容）
    ├── AudioDeviceError     音频设备错误（设备占用/不支持采样率/初始化失败）
    ├── InferenceError       推理运行时错误（CUDA OOM/张量形状不匹配）
    ├── ConfigError          配置错误（无效参数值/JSON 损坏）
    └── FeatureExtractError  特征提取错误（F0/HuBERT 提取失败）
"""


class RVCError(Exception):
    """所有 RVC 错误的基类。"""


class ModelLoadError(RVCError):
    """模型加载失败。"""


class AudioDeviceError(RVCError):
    """音频设备错误。"""


class InferenceError(RVCError):
    """推理运行时错误。"""


class ConfigError(RVCError):
    """配置错误。"""


class FeatureExtractError(RVCError):
    """特征提取错误。"""


def format_error_message(e: Exception) -> str:
    """把异常格式化为用户可读的消息（去除 traceback，保留关键信息）。

    用于 GUI 错误提示，避免向用户展示原始 Python 异常栈。
    """
    if isinstance(e, RVCError):
        return str(e)
    # 常见错误类型的友好消息
    msg = str(e)
    if "CUDA out of memory" in msg or "CUDA error: out of memory" in msg:
        return "显存不足，请降低 extra_time 或关闭其他占用显存的程序"
    if "No device" in msg or "Device unavailable" in msg:
        return "音频设备不可用，请检查设备是否被其他程序占用"
    if "Invalid sample rate" in msg:
        return "不支持的采样率，请尝试切换采样率模式"
    return msg
