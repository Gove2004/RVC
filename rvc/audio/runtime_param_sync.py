"""运行时参数同步器 — 线程安全的推理参数管理。

从 RealtimeEngine 中拆分出的参数管理组件，负责：
- 线程安全的 InferenceConfig 存储与访问
- 参数变更通知（回调机制，GUI 调参时触发）
- 参数快照（推理回调中读取，避免 GUI 线程修改导致的竞态）

RealtimeEngine 持有一个 RuntimeParamSync 实例，推理时通过 snapshot() 获取
当前参数快照，GUI 层通过 update() 修改参数。
"""
import copy
import threading
from typing import Callable

from rvc.core.config import InferenceConfig


class RuntimeParamSync:
    """运行时参数同步器 — 线程安全的推理参数管理。"""

    def __init__(self, initial_params: InferenceConfig | None = None):
        self._params = initial_params or InferenceConfig()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[InferenceConfig], None]] = []

    @property
    def params(self) -> InferenceConfig:
        """当前参数（直接引用，非线程安全。仅单线程场景使用）。"""
        return self._params

    def snapshot(self) -> InferenceConfig:
        """获取参数快照（深拷贝，线程安全）。

        推理回调中调用，避免 GUI 线程修改参数时导致的竞态。
        """
        with self._lock:
            return copy.deepcopy(self._params)

    def update(self, **kwargs) -> None:
        """批量更新参数（线程安全，触发变更通知）。

        Args:
            **kwargs: 要更新的参数字段，如 pitch=12, rms_mix=0.5
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
            snapshot = copy.deepcopy(self._params)
        self._notify(snapshot)

    def update_nested(self, field: str, **kwargs) -> None:
        """更新嵌套配置对象（如 denoise / break_protect）。

        Args:
            field: 嵌套字段名（"denoise" 或 "break_protect"）
            **kwargs: 要更新的子字段
        """
        with self._lock:
            nested = getattr(self._params, field)
            for key, value in kwargs.items():
                if hasattr(nested, key):
                    setattr(nested, key, value)
            snapshot = copy.deepcopy(self._params)
        self._notify(snapshot)

    def replace(self, params: InferenceConfig) -> None:
        """整体替换参数（线程安全，触发变更通知）。"""
        with self._lock:
            self._params = params
            snapshot = copy.deepcopy(self._params)
        self._notify(snapshot)

    def add_listener(self, callback: Callable[[InferenceConfig], None]) -> None:
        """注册参数变更监听器。

        Args:
            callback: 参数变更时调用，接收新的参数快照
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[InferenceConfig], None]) -> None:
        """移除参数变更监听器。"""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, snapshot: InferenceConfig) -> None:
        """通知所有监听器参数已变更。"""
        for callback in self._listeners:
            try:
                callback(snapshot)
            except Exception:
                pass  # 监听器异常不影响参数更新
