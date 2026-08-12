"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。"""
import logging
from dataclasses import dataclass

from rvc.audio import RealtimeEngine, get_audio_devices
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
    delay_ms: int


class InferController:
    def __init__(self, runtime_params=None, engine=None, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params or Params()
        self.inference_cache = inference_cache or default_inference_cache
        self.engine = engine or RealtimeEngine(self.runtime_params, self.inference_cache, on_runtime_error=on_runtime_error)

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
            use_pv=False,
            enable_out2=config.enable_out2,
            rms_mix=config.rms_mix,
            nr_enable=config.nr_enable,
            nr_strength=config.nr_strength,
        )

    def setup_engine(self, config: EngineConfig):
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
        delay = (self.engine.stream.latency[-1] if self.engine.stream else 0) + config.block_time + config.crossfade_time + 0.01
        return EngineStats(self.engine.sr_model, self.engine.sr_dev, int(delay * 1000))

    def stop(self):
        self.engine.stop()
