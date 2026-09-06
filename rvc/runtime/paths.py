"""运行时路径 — assets 资产路径统一管理。

所有 assets 下的路径只在本文件声明一次，各模块从这里取，禁止在脚本里硬编码。
路径一律基于项目根（PROJECT_ROOT）解析为绝对路径，与启动时 cwd 无关；
云训练（autodl_train.py）clone 后结构一致，同样可用。
"""
from pathlib import Path

# 项目根：rvc/runtime/paths.py → 向上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = PROJECT_ROOT / "assets"

# ── 运行时配置（configs 随仓库跟踪，仅 save_state.json 忽略）──
CONFIG_ROOT = ASSETS_ROOT / "configs"
STATE_FILE = CONFIG_ROOT / "save_state.json"

# 训练采样率 → 配置文件（40k/48k）
TRAIN_CONFIG_FILES = {
    40000: CONFIG_ROOT / "40ktrain_config.json",
    48000: CONFIG_ROOT / "48ktrain_config.json",
}
DEFAULT_TRAIN_SR = 48000

# ── 权重资产（大文件，git 忽略，目录骨架保留）──
# HuBERT 特征器：assets/hubert/{base,chinese}/ 各含三件套（config.json + preprocessor_config.json + pytorch_model.bin）
HUBERT_ROOT = ASSETS_ROOT / "hubert"
# F0 提取
RMVPE_PATH = ASSETS_ROOT / "rmvpe" / "rmvpe.pt"
# 预训练 G/D 底模：f0G/f0D{sr}k.pth
PRETRAINED_ROOT = ASSETS_ROOT / "pretrained"
# 训练导出的可用模型
MODELS_DIR = ASSETS_ROOT / "models"
# 本地训练工作目录（切片/特征/ckpt/导出中间产物），基于项目根、与启动 cwd 无关
TRAIN_LOGS_ROOT = PROJECT_ROOT / "logs"
# 人声提纯权重（手动下载；yaml 清单随仓库跟踪）
SEPARATE_DIR = ASSETS_ROOT / "separate"

# ── ffmpeg（Windows 专用二进制，git 忽略）──
FFMPEG_EXE = ASSETS_ROOT / "ffmpeg" / "ffmpeg.exe"
FFPROBE_EXE = ASSETS_ROOT / "ffmpeg" / "ffprobe.exe"

# ── 程序资源（png 随仓库跟踪）──
RESOURCES_DIR = ASSETS_ROOT / "resources"
ICON_IDLE_PATH = RESOURCES_DIR / "icon_idle.png"      # 红：未运行
ICON_ACTIVE_PATH = RESOURCES_DIR / "icon_active.png"  # 绿：推理中


def config_path() -> Path:
    return STATE_FILE


def parse_sr(sr) -> int:
    """把 '40k'/'48000' 等统一成 Hz 整数。

    采样率解析的唯一入口：训练/推理/人声提纯各处都走这里，避免重复实现。
    """
    s = str(sr).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))


def train_config_path(sr=None) -> Path:
    """按采样率返回训练配置路径；sr 缺省或不受支持时回退到默认 48k。"""
    if sr is not None:
        path = TRAIN_CONFIG_FILES.get(parse_sr(sr))
        if path is not None and path.exists():
            return path
    path = TRAIN_CONFIG_FILES[DEFAULT_TRAIN_SR]
    if not path.exists():
        raise FileNotFoundError(f"找不到训练配置: {path}")
    return path
