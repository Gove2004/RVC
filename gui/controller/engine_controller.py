"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。

- runtime_params: rvc.core.config.InferenceParams（推理参数）
- 引擎 setup 所需设备索引由调用方传入（设备名称→索引的转换在 window/device_catalog 层完成）

GUI 分层后，本类承担业务逻辑（参数应用、引擎控制、错误处理），
window.py 承担 UI 构建和信号连接，通过调用本类方法执行业务操作。
"""
import logging
import threading
from dataclasses import dataclass

from rvc.core.config import InferenceParams
from rvc.pipeline.cache import default_inference_cache
from rvc.streaming.passthrough import PassthroughStream
from gui.controller.telemetry import RuntimeTelemetry

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """引擎启动后的两组采样率，GUI 用来刷新「模型 XXXX / 设备 XXXX」单选按钮。"""
    model_sr: int   # 模型原生采样率
    device_sr: int  # 输入设备默认采样率（sr_mode="device" 时按它开流）


class InferController:
    def __init__(self, runtime_params: InferenceParams | None = None,
                 engine=None, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params or InferenceParams()
        self.inference_cache = inference_cache or default_inference_cache
        self._engine = engine  # None 时惰性构造（首次访问 self.engine 才加载 torch）
        self._engine_lock = threading.Lock()  # 防预热线程与主线程并发构造双实例
        self.on_runtime_error = on_runtime_error
        # 原声直通流（无 torch 依赖，窗口启动阶段即可用）
        self._passthrough = PassthroughStream()
        # 加载状态由 window 层持有（_loading + LoadThread），本类不再存第二份

    @property
    def engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from rvc.streaming.engine import VoiceEngine
                    self._engine = VoiceEngine(
                        self.runtime_params, self.inference_cache,
                        on_runtime_error=self.on_runtime_error,
                    )
        return self._engine

    @property
    def is_running(self) -> bool:
        """引擎是否在运行。"""
        return self._engine is not None and self.engine.running

    @property
    def models_dir(self):
        """模型文件对话框的默认起始目录（View 层不直接 import rvc 路径常量）。"""
        from rvc.runtime.paths import MODELS_DIR
        return MODELS_DIR

    @property
    def tray_icon_paths(self):
        """托盘三态图标路径 (idle红/active绿/passthrough蓝)，View 层经此取数（不直接 import rvc 路径常量）。"""
        from rvc.runtime.paths import (
            ICON_ACTIVE_PATH, ICON_IDLE_PATH, ICON_PASS_THROUGH_PATH,
        )
        return ICON_IDLE_PATH, ICON_ACTIVE_PATH, ICON_PASS_THROUGH_PATH

    @property
    def volume_range(self) -> tuple[float, float]:
        """输出音量滑条量程 (min, max)，View 层经此取数（不直接 import rvc 常量）。"""
        from rvc.core.config import VOLUME_MAX, VOLUME_MIN
        return VOLUME_MIN, VOLUME_MAX

    def list_gpus(self) -> list[str]:
        """可用的 CUDA 显卡名称列表（索引 = 显卡编号）。

        惰性 import torch：本方法只在引擎预热线程完成（torch 已加载）后调用，
        不参与窗口构建，故不会拖慢启动。
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            logger.warning("枚举 CUDA 显卡失败", exc_info=True)
            return []

    def preload_last_model(self) -> None:
        """后台预加载上次使用的模型（引擎预热线程收尾时调用）。

        配置里存有上次模型路径时提前完成 torch.load + 搬 GPU，用户点「变声」
        时 load_model 命中同 pth 快路径直接返回，只剩 warmup + 开流。
        失败静默（只留日志）——预加载是纯优化，真实加载的错误由点击路径正常上报。
        """
        pth = (getattr(self.runtime_params, "model_path", "") or "").strip()
        if not pth:
            return
        try:
            self.engine.load_model(pth, hubert=self.runtime_params.hubert)
            logger.info("上次模型已后台预加载，点「变声」将直接就绪")
        except Exception:
            logger.warning("后台预加载模型失败（点击变声时将正常加载）：%s", pth, exc_info=True)

    def snapshot(self) -> RuntimeTelemetry:
        """运行状态只读快照（View 订阅快照，不穿透引擎内部）。"""
        eng = self._engine
        if eng is None or not eng.running:
            return RuntimeTelemetry(running=False, measure_ms=0.0, input_pitch=0.0)
        return RuntimeTelemetry(
            running=True,
            measure_ms=eng.measure_ms,
            input_pitch=eng.input_pitch,
            error_count=eng.error_count,
            driver_in_ms=eng.driver_in_ms,
            driver_out_ms=eng.driver_out_ms,
            infer_ms=eng.infer_ms,
        )

    # ── 引擎控制 ──

    def setup_engine(self, sr_mode: str, input_device_idx: int, output_device_idx: int,
                     output2_device_idx: int, block_time: float, crossfade_time: float,
                     extra_time: float, enable_out2: bool):
        """配置并启动音频引擎。

        设备索引由调用方（window/device_catalog）从设备名称转换而来。
        副输出由 setup 内部统一启动（setup_out2 是幂等的，重复调用安全）。
        """
        out2_idx = output2_device_idx if (enable_out2 and output2_device_idx >= 0) else None
        try:
            self.engine.setup(
                sr_mode, input_device_idx, output_device_idx,
                block_time, crossfade_time, extra_time, out2_idx,
            )
        except Exception:
            self.engine.stop()  # 启动失败时停掉所有流，避免引擎失控
            raise
        return EngineStats(self.engine.model_sr, self.engine.device_sr)

    def stop(self):
        """停止引擎。"""
        if self._engine is not None:
            self.engine.stop()

    # ── 原声直通 ──

    def start_passthrough(self, in_dev: int, out_dev: int) -> None:
        """原声直通：把麦克风原声转发到主输出设备（不经模型）。"""
        self._passthrough.start(in_dev, out_dev)

    def stop_passthrough(self) -> None:
        """关闭原声直通流（幂等）。"""
        self._passthrough.stop()

    @property
    def passthrough_running(self) -> bool:
        """直通流是否在运行。"""
        return self._passthrough.running

    # ── 错误处理 ──

    def handle_runtime_error(self, message: str) -> None:
        """处理运行时错误（由音频回调线程触发，通过信号转发到主线程后调用）。

        流已在音频回调内通过 CallbackStop 安全停止；这里只复位运行时状态。
        """
        if self._engine is not None:
            self.engine.runtime_error_pending = False
            self.engine.stop()
        logger.error("实时推理错误：%s", message)
