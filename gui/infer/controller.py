"""推理控制器 — 管理运行时参数、引擎启动和设备绑定。"""
from dataclasses import dataclass

from rvc.audio import RealtimeEngine, get_audio_devices
from rvc.models import default_inference_cache
from rvc.inference import Params


@dataclass
class ModelConfig:
    pitch: int
    index_rate: float
    rms_mix: float
    gender: float
    protect: float
    f0method: str


@dataclass
class RuntimeConfig:
    eq_en: bool
    eq_sub: float
    eq_low: float
    eq_mid: float
    eq_hi_mid: float
    eq_high: float
    reverb: float
    out2_enabled: bool


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
    def __init__(self, runtime_params=None, engine=None, inference_cache=None):
        self.runtime_params = runtime_params or Params()
        self.inference_cache = inference_cache or default_inference_cache
        self.engine = engine or RealtimeEngine(self.runtime_params, self.inference_cache)

    def apply_model_config(self, config: ModelConfig):
        self.runtime_params.update(
            pitch=config.pitch,
            index_rate=config.index_rate,
            rms_mix=config.rms_mix,
            gender=config.gender,
            protect=config.protect,
            f0method=config.f0method,
        )

    def apply_runtime_config(self, config: RuntimeConfig):
        self.runtime_params.update(
            use_pv=False,
            enable_eq=config.eq_en,
            eq_sub=config.eq_sub,
            eq_low=config.eq_low,
            eq_mid=config.eq_mid,
            eq_hi_mid=config.eq_hi_mid,
            eq_high=config.eq_high,
            reverb=config.reverb,
            bgm_enable=False,
            enable_out2=config.out2_enabled,
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
        self.engine.bgm_audio = None
        self.engine.bgm_ptr = 0
        if self.runtime_params.enable_out2:
            self.engine.setup_out2(out_idx[config.output2_device_pos])
        delay = (self.engine.stream.latency[-1] if self.engine.stream else 0) + config.block_time + config.crossfade_time + 0.01
        return EngineStats(self.engine.sr_model, self.engine.sr_dev, int(delay * 1000))

    def stop(self):
        self.engine.stop()
