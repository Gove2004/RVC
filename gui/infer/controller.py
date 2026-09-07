"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。

配置统一使用 rvc.config.AppConfig 的子配置：
- runtime_params: InferenceConfig（推理参数）
- engine setup: 由调用方传入设备索引（设备名称→索引的转换在 window/device_manager 层完成）
"""
import logging
import threading
from dataclasses import dataclass

from rvc.config import InferenceConfig
from rvc.models import default_inference_cache

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    sr_model: int
    sr_dev: int


class InferController:
    def __init__(self, runtime_params: InferenceConfig | None = None,
                 engine=None, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params or InferenceConfig()
        self.inference_cache = inference_cache or default_inference_cache
        self._engine = engine  # None 时惰性构造（首次访问 self.engine 才加载 torch）
        self._engine_lock = threading.Lock()  # 防预热线程与主线程并发构造双实例
        self.on_runtime_error = on_runtime_error

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

    def apply_model_params(self, pitch: int, formant: float, protect: float, f0_method: str):
        """应用模型卡片的推理参数（音高/音色/保护/F0方法）。"""
        self.runtime_params.pitch = pitch
        self.runtime_params.formant = formant
        self.runtime_params.protect = protect
        self.runtime_params.f0_method = f0_method

    def apply_runtime_params(self, rms_mix: float, enable_out2: bool,
                             denoise_enable: bool, denoise_strength: float,
                             break_enable: bool, break_src_hz: float):
        """应用运行时参数（响度混合/副输出/降噪/破音保护）。"""
        self.runtime_params.rms_mix = rms_mix
        self.runtime_params.denoise.enable = denoise_enable
        self.runtime_params.denoise.strength = denoise_strength
        self.runtime_params.break_protect.enable = break_enable
        self.runtime_params.break_protect.src_hz = break_src_hz
        # enable_out2 属于引擎配置，由 engine 直接读取，这里不设置

    def setup_engine(self, sr_mode: str, input_device_idx: int, output_device_idx: int,
                     output2_device_idx: int, block_time: float, crossfade_time: float,
                     extra_time: float, enable_out2: bool):
        """配置并启动音频引擎。

        设备索引由调用方（window/device_manager）从设备名称转换而来。
        """
        sr_type = "sr_model" if sr_mode == "model" else "sr_device"
        out2_idx = output2_device_idx if (enable_out2 and output2_device_idx >= 0) else None
        self.engine.setup(
            sr_type, input_device_idx, output_device_idx,
            block_time, crossfade_time, extra_time, out2_idx,
        )
        if enable_out2 and output2_device_idx >= 0:
            try:
                self.engine.setup_out2(output2_device_idx)
            except Exception:
                self.engine.stop()  # 副输出失败时停掉主流，避免引擎失控
                raise
        return EngineStats(self.engine.sr_model, self.engine.sr_dev)

    def stop(self):
        self.engine.stop()
