"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。

配置统一使用 rvc.config.AppConfig 的子配置：
- runtime_params: InferenceConfig（推理参数）
- engine setup: 由调用方传入设备索引（设备名称→索引的转换在 window/device_manager 层完成）

GUI 分层后，本类承担业务逻辑（参数应用、引擎控制、错误处理），
window.py 承担 UI 构建和信号连接，通过调用本类方法执行业务操作。
"""
import logging
import threading
from dataclasses import dataclass

from rvc.core.config import InferenceConfig
from rvc.inference.inference_cache import default_inference_cache

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    sr_model: int
    sr_dev: int


@dataclass
class StartResult:
    """开始推理的结果。"""
    success: bool
    error: str = ""
    sr_model: int = 0
    sr_dev: int = 0


class InferController:
    def __init__(self, runtime_params: InferenceConfig | None = None,
                 engine=None, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params or InferenceConfig()
        self.inference_cache = inference_cache or default_inference_cache
        self._engine = engine  # None 时惰性构造（首次访问 self.engine 才加载 torch）
        self._engine_lock = threading.Lock()  # 防预热线程与主线程并发构造双实例
        self.on_runtime_error = on_runtime_error
        # 加载线程状态（由 window 层管理 LoadThread，本类只提供业务回调）
        self._loading = False
        self._load_thread = None

    @property
    def engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from rvc.audio import RealtimeEngine
                    self._engine = RealtimeEngine(
                        self.runtime_params, self.inference_cache,
                        on_runtime_error=self.on_runtime_error,
                    )
        return self._engine

    @property
    def is_running(self) -> bool:
        """引擎是否在运行。"""
        return self._engine is not None and self.engine.running

    @property
    def is_loading(self) -> bool:
        """模型是否正在加载。"""
        return self._loading

    # ── 参数应用 ──

    def apply_model_params(self, pitch: int, formant: float):
        """应用模型卡片级参数（音高/共振峰/特征器）。"""
        self.runtime_params.pitch = pitch
        self.runtime_params.formant = formant

    def apply_runtime_params(self, protect: float, f0_method: str, rms_mix: float):
        """应用全局推理参数（辅音保护/F0方法/响度混合）。"""
        self.runtime_params.protect = protect
        self.runtime_params.f0_method = f0_method
        self.runtime_params.rms_mix = rms_mix

    # ── 引擎控制 ──

    def setup_engine(self, sr_mode: str, input_device_idx: int, output_device_idx: int,
                     output2_device_idx: int, block_time: float, crossfade_time: float,
                     extra_time: float, enable_out2: bool):
        """配置并启动音频引擎。

        设备索引由调用方（window/device_manager）从设备名称转换而来。
        副输出由 setup 内部统一启动（setup_out2 是幂等的，重复调用安全）。
        """
        sr_type = "sr_model" if sr_mode == "model" else "sr_device"
        out2_idx = output2_device_idx if (enable_out2 and output2_device_idx >= 0) else None
        try:
            self.engine.setup(
                sr_type, input_device_idx, output_device_idx,
                block_time, crossfade_time, extra_time, out2_idx,
            )
        except Exception:
            self.engine.stop()  # 启动失败时停掉所有流，避免引擎失控
            raise
        return EngineStats(self.engine.sr_model, self.engine.sr_dev)

    def start_inference(self, pth_path: str, hubert: str,
                        sr_mode: str, input_device_idx: int, output_device_idx: int,
                        output2_device_idx: int, block_time: float, crossfade_time: float,
                        extra_time: float, enable_out2: bool) -> StartResult:
        """同步开始推理（加载模型 + 启动引擎）。

        用于离线推理或测试场景。GUI 场景使用异步加载（LoadThread）。

        Returns:
            StartResult: 包含成功标志、错误信息和采样率信息
        """
        try:
            self.engine.load_model(pth_path, hubert=hubert)
            stats = self.setup_engine(
                sr_mode, input_device_idx, output_device_idx,
                output2_device_idx, block_time, crossfade_time, extra_time, enable_out2,
            )
            return StartResult(success=True, sr_model=stats.sr_model, sr_dev=stats.sr_dev)
        except Exception as e:
            logger.error("开始推理失败：%s", e, exc_info=True)
            self.stop()
            return StartResult(success=False, error=str(e))

    def stop(self):
        """停止引擎。"""
        if self._engine is not None:
            self.engine.stop()

    def stop_inference(self) -> None:
        """停止推理（停止引擎 + 重置加载状态）。"""
        self.stop()
        self._loading = False
        if self._load_thread is not None:
            self._load_thread = None

    # ── 加载线程管理 ──

    def begin_load(self, load_thread) -> None:
        """标记开始加载模型（由 window 层创建 LoadThread 后调用）。"""
        self._loading = True
        self._load_thread = load_thread

    def end_load(self) -> None:
        """标记加载结束（LoadThread finished 时调用）。"""
        self._loading = False
        self._load_thread = None

    def cancel_load(self, timeout: int = 3000) -> bool:
        """取消正在进行的加载（请求停止 + 等待超时）。

        Returns:
            True 表示线程已结束，False 表示超时
        """
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.request_stop()
            return self._load_thread.wait(timeout)
        return True

    # ── 错误处理 ──

    def handle_runtime_error(self, message: str) -> None:
        """处理运行时错误（由音频回调线程触发，通过信号转发到主线程后调用）。

        流已在音频回调内通过 CallbackStop 安全停止；这里只复位运行时状态。
        """
        if self._engine is not None:
            self.engine.runtime_error_pending = False
            self.engine.stop()
        logger.error("实时推理错误：%s", message)
