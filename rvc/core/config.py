"""统一配置体系 — 训练/推理/GUI 共用一套 dataclass。"""
from dataclasses import dataclass, field

HUBERT_DEFAULT = "chinese"


@dataclass
class InferenceConfig:
    formant: float = 0.0
    protect: float = 0.5
    f0_method: str = "rmvpe"
    rms_mix: float = 0.0


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
    formant: float = 0.0
    hubert: str = "chinese"


@dataclass
class AppConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    active_model: str = ""
    models: list[ModelEntry] = field(default_factory=list)


@dataclass
class OfflineConfig(InferenceConfig):
    """离线推理任务配置 = 音频效果参数（继承 InferenceConfig）+ 路径/模型信息。"""
    input_path: str = ""
    output_path: str = ""
    model_path: str = ""
    hubert: str = HUBERT_DEFAULT


@dataclass
class TrainConfig:
    """训练运行时配置——传给 Trainer 的超参数（epochs/batch_size/lr 等）。

    与 gui.configs.train_state.TrainGuiState 的区别：
    - TrainConfig: 运行时配置，直接控制训练循环，不含路径/GUI 状态
    - TrainGuiState: GUI 状态持久化，含 exp_name/input_dir 等路径信息
    """
    exp_dir: str
    sr: int = 48000
    epochs: int = 2000
    batch_size: int = 4
    save_every_epoch: int = 200
    learning_rate: float = 1e-4
    pretrain_g: str = ""
    pretrain_d: str = ""
    fp16_run: bool = True
    device: str = "cuda:0"
    log_interval: int = 20
    keep_ckpts: int = 1
    keep_models: int = 0

