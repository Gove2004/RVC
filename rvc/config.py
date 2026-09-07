"""统一配置体系 — 训练/推理/GUI 共用一套 dataclass。"""
from dataclasses import dataclass, field

HUBERT_DEFAULT = "chinese"


@dataclass
class BreakProtectConfig:
    enable: bool = True
    src_hz: float = 300.0
    ratio: float = 0.4
    knee: float = 0.12


@dataclass
class DenoiseConfig:
    enable: bool = False
    strength: float = 0.5


@dataclass
class InferenceConfig:
    pitch: int = 0
    formant: float = 0.0
    protect: float = 0.5
    f0_method: str = "rmvpe"
    rms_mix: float = 0.0
    hubert_variant: str = "chinese"
    break_protect: BreakProtectConfig = field(default_factory=BreakProtectConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class EngineConfig:
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    sr_mode: str = "model"
    hostapi: str = ""
    input_device: str = ""
    output_device: str = ""
    output2_device: str = ""
    enable_out2: bool = False


@dataclass
class ModelEntry:
    name: str = ""
    pth: str = ""
    pitch: int = 0
    formant: float = 0.0
    hubert: str = "chinese"


@dataclass
class AppConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    active_model: str = ""
    models: list[ModelEntry] = field(default_factory=list)
