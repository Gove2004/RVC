"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。"""
import logging
import threading
from dataclasses import dataclass

# 注意：RealtimeEngine 依赖 torch，惰性构造（见 self.engine property），
# 避免 GUI 窗口出现前就加载重型依赖。
from rvc.models import default_inference_cache
from rvc.inference import Params

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    pitch: int
    index_rate: float
    gender: float
    protect: float
    f0method: str


@dataclass
class RuntimeConfig:
    enable_out2: bool
    rms_mix: float
    nr_enable: bool = False
    nr_strength: float = 0.5
    break_enable: bool = True
    break_src_hz: float = 300.0


@dataclass
class EngineConfig:
    hostapi_name: str
    input_device_pos: int
    output_device_pos: int
    output2_device_pos: int
    sr_mode: str
    block_time: float
    crossfade_time: float
    extra_time: float


@dataclass
class EngineStats:
    sr_model: int
    sr_dev: int


class InferController:
    def __init__(self, runtime_params=None, engine=None, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params or Params()
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
                        self.runtime_params, self.inference_cache, on_runtime_error=self.on_runtime_error
                    )
        return self._engine

    def apply_model_config(self, config: ModelConfig):
        self.runtime_params.update(
            pitch=config.pitch,
            index_rate=config.index_rate,
            gender=config.gender,
            protect=config.protect,
            f0method=config.f0method,
        )

    def apply_runtime_config(self, config: RuntimeConfig):
        self.runtime_params.update(
            enable_out2=config.enable_out2,
            rms_mix=config.rms_mix,
            nr_enable=config.nr_enable,
            nr_strength=config.nr_strength,
            break_enable=config.break_enable,
            break_src_hz=config.break_src_hz,
        )

    def setup_engine(self, config: EngineConfig):
        from rvc.audio import get_audio_devices  # 惰性导入（device_query，轻量）

        _, _, _, in_idx, out_idx = get_audio_devices(config.hostapi_name)
        sr_type = "sr_model" if config.sr_mode == "model" else "sr_device"
        self.engine.setup(
            sr_type,
            in_idx[config.input_device_pos],
            out_idx[config.output_device_pos],
            config.block_time,
            config.crossfade_time,
            config.extra_time,
        )
        if self.runtime_params.enable_out2 and config.output2_device_pos >= 0:
            try:
                self.engine.setup_out2(out_idx[config.output2_device_pos])
            except Exception:
                self.engine.stop()  # 副输出失败时停掉主流，避免引擎失控
                raise
        # 延迟显示已改用硬件时间戳实测（engine.measure_ms，见 _cb），此处不再估算。
        return EngineStats(self.engine.sr_model, self.engine.sr_dev)

    def stop(self):
        self.engine.stop()
