"""人声提纯模型注册表。

上游 pymss 收录了 319 个模型，这里只保留训练素材提纯链路实际用到的 5 个，
避免把整套目录搬进来。model_type 决定走哪套架构代码：
  htdemucs / bs_roformer → 走 demix.py 的分块推理（需要 YAML config）
  vr                     → 走 vocal_remover/vr_separator（权重内含结构，无需 config）

stem 是要保留的那一路输出名。对 target_instrument 非空的模型（bs317 / dereverb /
karaoke），模型本身只输出一路，stem 仅用于校验；htdemucs 输出四路，靠 stem 挑人声。
"""
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = _PROJECT_ROOT / "assets" / "separate"

MS_REPO = "baicai1145/pymss"
HF_REPO = "baicai1145/pymss"

# ModelScope 国内直连通常比 HF 快得多，作为默认源
MS_BASE_URL = f"https://www.modelscope.cn/models/{MS_REPO}/resolve/master"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"
HF_MIRROR_URL = f"https://hf-mirror.com/{HF_REPO}/resolve/main"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".wv", ".aif", ".aiff"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    relpath: str
    config_relpath: str
    model_type: str
    stem: str
    size_mb: float
    note: str = ""

    @property
    def weight_path(self) -> Path:
        return MODEL_DIR / self.filename

    @property
    def config_path(self) -> Path | None:
        if not self.config_relpath:
            return None
        return MODEL_DIR / Path(self.config_relpath).name

    @property
    def is_ready(self) -> bool:
        if not self.weight_path.exists():
            return False
        return self.config_path is None or self.config_path.exists()


# key → spec。GUI 按下拉顺序展示。
MODELS: dict[str, ModelSpec] = {
    "htdemucs": ModelSpec(
        key="htdemucs",
        label="HTDemucs v4（快 · 84MB）",
        filename="HTDemucs4.th",
        relpath="music_stems/four_stem/HTDemucs4.th",
        config_relpath="music_stems/four_stem/HTDemucs4.yaml",
        model_type="htdemucs",
        stem="vocals",
        size_mb=84.1,
        note="Demucs v4 四路分离。速度最快、显存最省，日常批量首选。",
    ),
    "bs317": ModelSpec(
        key="bs317",
        label="BS-Roformer 317（强 · 639MB）",
        filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        relpath="vocal/vocal_extraction/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        config_relpath="vocal/vocal_extraction/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        model_type="bs_roformer",
        stem="Vocals",
        size_mb=639.3,
        note="SDR 12.98，人声最干净、伴奏残留最少。慢且吃显存，定稿素材用它。",
    ),
    "dereverb": ModelSpec(
        key="dereverb",
        label="去混响（204MB）",
        filename="dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        relpath="reverb_echo_control/dereverb/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        config_relpath="reverb_echo_control/dereverb/dereverb_bs_roformer_anvuew_sdr_22.5050.yaml",
        model_type="bs_roformer",
        stem="noreverb",
        size_mb=204.5,
        note="SDR 22.51。混响是训练素材头号杀手，训出来会有金属味和拖尾。",
    ),
    "karaoke": ModelSpec(
        key="karaoke",
        label="去和声（204MB）",
        filename="bs_roformer_karaoke_anvuew.ckpt",
        relpath="karaoke/bs_roformer_karaoke_anvuew.ckpt",
        config_relpath="karaoke/bs_roformer_karaoke_anvuew.yaml",
        model_type="bs_roformer",
        stem="Vocals",
        size_mb=204.5,
        note="剥离伴唱和声，只留主唱。翻唱/卡拉OK 素材才需要。",
    ),
    "denoise": ModelSpec(
        key="denoise",
        label="去杂音（127MB）",
        filename="UVR-DeNoise.pth",
        relpath="legacy_vr/vr_denoise/UVR-DeNoise.pth",
        config_relpath="",  # VR 架构权重自带结构，用内置 vr_modelparams，不需要 YAML
        model_type="vr",
        stem="No Noise",
        size_mb=127.1,
        note="VR 架构去噪：底噪、环境噪声、电流声。输出 No Noise 那一路。",
    ),
}

# 主分离模型（二选一，互斥）
EXTRACT_KEYS = ("htdemucs", "bs317")
# 可选后处理（可叠加，按此顺序执行）
POST_KEYS = ("dereverb", "karaoke", "denoise")


def get(key: str) -> ModelSpec:
    return MODELS[key]


def missing(keys) -> list[ModelSpec]:
    """返回还没有下载好的模型 spec 列表。"""
    return [MODELS[k] for k in keys if not MODELS[k].is_ready]
