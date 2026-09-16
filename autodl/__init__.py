"""RVC 云 GPU 训练向导 — 共享常量、异常与路径初始化。"""
import os
import sys
from pathlib import Path

from rvc.core.config import HUBERT_DEFAULT, TrainConfig

# 项目内大量路径是相对 cwd 的，统一以脚本所在目录（=项目根）为基准。
# 切 cwd 的动作在 main() 第一行做（Q6/D4：import 即 chdir 属隐式副作用），
# 这里只保留 import 解析所需的 sys.path 注入。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 云上系统盘只有 30G，训练产物一律走数据盘；本地降级到项目内
_DEFAULT_CLOUD_ROOT = Path("/root/autodl-tmp")
if os.environ.get("RVC_OUT_ROOT", "").strip():
    DATA_ROOT = Path(os.environ["RVC_OUT_ROOT"].strip()).expanduser()
elif _DEFAULT_CLOUD_ROOT.is_dir():
    DATA_ROOT = _DEFAULT_CLOUD_ROOT
else:
    DATA_ROOT = PROJECT_ROOT / "autodl-tmp"
LOGS_ROOT = DATA_ROOT / "logs"
MODELS_ROOT = DATA_ROOT / "models"

# 环境检测用（纯元数据查询，不真正执行模块）
REQUIRED_PACKAGES = [
    ("torch", "torch", "PyTorch 深度学习框架（训练核心）"),
    ("numpy", "numpy", "数组计算"),
    ("scipy", "scipy", "滤波器 / 重采样"),
    ("transformers", "transformers", "HuBERT 特征提取"),
]
OPTIONAL_PACKAGES = [
    ("PySide6", "PySide6", "GUI（云端不需要）"),
    ("sounddevice", "sounddevice", "实时音频 IO（云端不需要）"),
]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
NATIVE_EXTS = {".wav", ".flac", ".ogg"}

EXIT_OK, EXIT_ENV, EXIT_RUNTIME = 0, 1, 2
BAR = "─" * 66

# 向导交互默认值（D6 单源：能对上 dataclass 的项从 rvc.core.config 取；
# epochs/sr 展示默认与训练配置字面不同，属 UI 展示现状，保留字面量并注明）
# TrainConfig.exp_dir 为必填项，不可实例化，故直接读字段默认值
_TC_FIELD = TrainConfig.__dataclass_fields__
DEFAULTS = {
    "exp": "test",
    "sr": "48k",
    "epochs": 250,  # 向导展示默认（与 TrainConfig.epochs=2000 / GUI 200 的差异为展示现状）
    "lr": _TC_FIELD["learning_rate"].default,
    "save_every": 20,
    "keep_ckpts": _TC_FIELD["keep_ckpts"].default,
    "per": 3.7,
    "keep_models": _TC_FIELD["keep_models"].default,
    "hubert": HUBERT_DEFAULT,
    "use_pretrained": True,
}


class Cancelled(Exception):
    """用户 Ctrl+C 取消。"""


class EnvFatal(Exception):
    """环境体检致命项缺失。"""
