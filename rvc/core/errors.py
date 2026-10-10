"""RVC 统一异常体系。

所有核心模块抛出自定义异常，GUI 层按类型做差异化处理：
- ModelLoadError → 弹窗提示 + 保留旧模型
- AudioDeviceError → 提示切换设备
- InferenceError → 停止推理 + 显示错误
"""


class RVCError(Exception):
    """所有 RVC 异常的基类。"""


class ModelLoadError(RVCError):
    """模型加载失败（pth 损坏/缺失/不兼容）。"""


class AudioDeviceError(RVCError):
    """音频设备错误（设备不存在/不支持通道/打开失败）。"""


class InferenceError(RVCError):
    """推理过程错误（合成器/F0/特征提取失败）。"""


class F0ExtractionError(InferenceError):
    """F0 提取失败（RMVPE/FCPE 权重缺失/推理异常）。"""


class FeatureExtractionError(InferenceError):
    """HuBERT 特征提取失败。"""


class AudioLoadError(RVCError):
    """音频加载失败（文件不存在/解码失败/格式不支持）。"""


class CancelledError(BaseException):
    """离线推理被用户取消时抛出。

    刻意**不**继承 RVCError/Exception：取消是用户意图而非错误。
    与标准库 asyncio.CancelledError 同口径——若继承 Exception，任何
    「except Exception」兜底（日志/弹窗/重试）都会把取消误判为故障，
    并吞掉取消信号让上层的取消分支永不触发。调用方必须显式
    「except CancelledError」捕获（见 gui/controller/offline_controller.py）。
    """
