"""统一配置体系 — 训练/推理/GUI 共用一套 dataclass。"""
from dataclasses import dataclass, field

HUBERT_DEFAULT = "chinese"


@dataclass
class InferenceConfig:
    """推理参数 — 音频效果 + 辅音保护 + 音域映射 + F0 提取阈值。

    原 experimental_config 中的核心参数已合并到此，统一管理。
    """
    # ── 基础音频效果 ──
    formant: float = 0.0
    protect: float = 0.5
    f0_method: str = "rmvpe"
    rms_mix: float = 0.0
    # ── 辅音保护（渐变式，始终生效） ──
    rmvpe_threshold: float = 0.05  # RMVPE 清浊判定阈值
    fcpe_confidence_threshold: float = 0.05  # FCPE 清浊判定阈值
    protect_soft_threshold_hz: float = 20.0  # 过渡中心（Hz）
    protect_soft_width: float = 30.0  # 过渡宽度（Hz，sigmoid 的 4σ）
    # ── 音域映射（半音尺度，始终生效） ──
    pitch_map_src_min: float = 100.0   # 原声音域下限 (Hz)
    pitch_map_src_max: float = 500.0   # 原声音域上限 (Hz)
    pitch_map_dst_min: float = 200.0   # 目标音域下限 (Hz)
    pitch_map_dst_max: float = 800.0   # 目标音域上限 (Hz)


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
class AppConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    model_path: str = ""
    hubert: str = HUBERT_DEFAULT


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

