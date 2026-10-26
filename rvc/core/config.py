"""统一配置体系 — 训练/推理/GUI 共用一套 dataclass。

推理参数统一为 InferenceParams，按语义分组：
- voice: 音色相关（formant、音域映射）
- f0: 基频提取相关（方法、阈值）
- buffer: 缓冲区相关（block/crossfade/extra）
- audio: 音频设备相关（采样率、设备）
- realism: 真实感（参考输入的底噪/气息/近讲/距离搬运强度）
- 顶层: rms_mix、model_path、hubert
"""
from dataclasses import dataclass, field

from rvc.core.constants import (
    FCPE_CONFIDENCE_THRESHOLD_INFER,
    RMVPE_THRESHOLD_INFER,
)

# HuBERT 特征器默认变体。训练与推理必须用同一个（assets/hubert/{base,chinese}）：
# 不一致时音色发闷、咬字怪。改这里即全链路默认切换。
HUBERT_DEFAULT = "chinese"

# 输出音量增益范围（唯一事实源：GUI 滑动条量程与热路径钳位都取这里）
VOLUME_MIN = 0.0
VOLUME_MAX = 2.5

# 性别因子（formant）量程，单位半音。GUI 滑条量程与实时启动时的图预捕获范围
# 都取这里——两处若各写一份，滑条改宽了而预捕获没跟上，超出部分会在音频回调
# 线程里现场捕获（约 0.25s 断音）。
FORMANT_MIN = -2.5
FORMANT_MAX = 2.5


@dataclass
class VoiceParams:
    """音色参数 — 共振峰 + 音域映射。"""
    formant: float = 0.0
    # 音域映射（Hz，始终生效）
    pitch_map_src_min: float = 100.0
    pitch_map_src_max: float = 500.0
    pitch_map_dst_min: float = 200.0
    pitch_map_dst_max: float = 800.0


@dataclass
class F0Params:
    """基频提取参数 — 方法 + 清浊判定阈值。

    阈值默认值单源取自 rvc.core.constants（推理侧常量），本文件不写字面量。
    """
    method: str = "rmvpe"  # rmvpe / fcpe
    rmvpe_threshold: float = RMVPE_THRESHOLD_INFER
    fcpe_confidence_threshold: float = FCPE_CONFIDENCE_THRESHOLD_INFER


@dataclass
class BufferParams:
    """缓冲区参数 — 控制实时推理延迟和音质。"""
    block_time: float = 0.25      # 每块时长（秒）
    crossfade_time: float = 0.05  # 交叉淡化时长（秒；SOLA 上限 50ms）
    extra_time: float = 2.5        # 额外上下文（秒）


# 真实感四级到单滑条的配比（权重递减）。顺序即「先开哪个」：
# residual/breath 只在输入本身有底噪/齿音时才注入，最安全；distance 让电平
# 自己起伏，最容易被误当成 bug，所以排在最后、权重最小。
REALISM_MIX = (("residual", 1.0), ("breath", 0.85), ("rumble", 0.7), ("distance", 0.55))


@dataclass
class RealismParams:
    """真实感参数 — 把参考输入的录音痕迹搬到输出上的四级强度（0 = 逐位旁路）。

    四级互相不可替代，逐级对应一种「不像录音」：residual（非谐波底噪）、
    breath（>4kHz 气息/齿音）、rumble（近讲低频）、distance（慢速远近起伏）。
    默认全 0：每一级都是叠加型染色，开大就从「更真实」变成「加特效」。
    """
    breath: float = 0.0
    residual: float = 0.0
    rumble: float = 0.0
    distance: float = 0.0

    @property
    def strength(self) -> float:
        """单滑条强度（0~1）— 按配比折算后取最大值，保证 set/get 往返一致。"""
        return max(getattr(self, name) / w for name, w in REALISM_MIX)

    @strength.setter
    def strength(self, value: float) -> None:
        s = min(max(float(value), 0.0), 1.0)
        for name, w in REALISM_MIX:
            setattr(self, name, s * w)


@dataclass
class AudioParams:
    """音频设备参数 — 采样率模式 + 设备选择。"""
    sr_mode: str = "model"  # model / device
    hostapi: str = ""
    input_device: str = ""
    output_device: str = ""
    output2_device: str = ""
    enable_out2: bool = False
    # 推理所用 CUDA 设备编号（多卡机器选卡）。进程启动时经 RuntimeDeviceConfig
    # 生效一次；改动后需重启程序（模型与 CUDA Graph 已按旧设备捕获/驻留）。
    device_index: int = 0


# InferenceParams 的嵌套分组清单（序列化 / 参数拷贝共用一份）
PARAM_GROUPS = ("voice", "f0", "buffer", "audio", "realism")


@dataclass
class InferenceParams:
    """统一推理参数 — 同时承担持久化配置和运行时参数职责。

    滑动条直接绑定此对象的字段，valueChanged 时自动更新。
    字段访问使用分组路径：params.voice.formant、params.f0.method 等。
    """
    voice: VoiceParams = field(default_factory=VoiceParams)
    f0: F0Params = field(default_factory=F0Params)
    buffer: BufferParams = field(default_factory=BufferParams)
    audio: AudioParams = field(default_factory=AudioParams)
    realism: RealismParams = field(default_factory=RealismParams)
    rms_mix: float = 0.0
    volume: float = 1.0  # 输出音量增益（0.0-2.5，1.0=原始音量，纯标量乘法）
    model_path: str = ""
    hubert: str = HUBERT_DEFAULT

    def update_from(self, other: "InferenceParams") -> None:
        """原地复制另一个 InferenceParams 的所有字段（包括嵌套分组）。

        不替换对象引用，只修改字段值。这样所有持有此对象引用的地方
        （engine/runner/pipeline）都会自动看到更新后的值，无需同步。
        """
        # 嵌套分组：复制每个分组的所有字段
        for group_name in PARAM_GROUPS:
            src_group = getattr(other, group_name)
            dst_group = getattr(self, group_name)
            for field_name in src_group.__dataclass_fields__:
                setattr(dst_group, field_name, getattr(src_group, field_name))
        # 顶层字段
        self.rms_mix = other.rms_mix
        self.volume = other.volume
        self.model_path = other.model_path
        self.hubert = other.hubert


@dataclass
class OfflineParams(InferenceParams):
    """离线推理任务配置 = 继承 InferenceParams 所有音频参数 + 路径信息。"""

    input_path: str = ""
    output_path: str = ""

    @classmethod
    def from_inference(cls, src: InferenceParams, *, input_path: str,
                       output_path: str, hubert: str | None = None) -> "OfflineParams":
        """从 GUI 收集的 InferenceParams 整体派生离线任务参数。

        必须经此构造，禁止逐字段手抄——手抄在新增参数时必然被遗漏
        （protect 漏传导致离线恒 0 的事故）。update_from 是唯一拷贝路径，
        以后给 InferenceParams 加任何字段，离线自动带上。
        """
        cfg = cls(input_path=input_path, output_path=output_path)
        cfg.update_from(src)
        if hubert is not None:
            cfg.hubert = hubert
        return cfg


@dataclass
class TrainConfig:
    """训练运行时配置——传给 Trainer 的超参数（epochs/batch_size/lr 等）。

    仅本 dataclass 覆盖的字段可在此调整；训练底参（c_mel/c_kl/lr_decay/
    seed/segment_size/betas/eps）不在 Python 里定义，唯一来源是
    assets/configs/<40k|48k>train_config.json（按采样率二选一）。
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
    keep_ckpts: int = 1
    keep_models: int = 0
