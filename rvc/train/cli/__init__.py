"""训练命令行向导 — 共享常量、异常与产物根目录解析。

Windows 与 Linux 共用同一套纯命令行流程。本包只依赖 `rvc.*`，不依赖 GUI。

入口：`python app.py --train` 或 `python -m rvc.train.cli`。
"""
import os
from pathlib import Path

from rvc.core.config import HUBERT_DEFAULT, TrainConfig
from rvc.core.constants import AUDIO_EXTS, NATIVE_EXTS, SLICE_PERIOD_DEFAULT
from rvc.runtime.paths import PROJECT_ROOT

# 环境检测用（纯元数据查询，不真正执行模块）
REQUIRED_PACKAGES = [
    ("torch", "torch", "PyTorch 深度学习框架（训练核心）"),
    ("numpy", "numpy", "数组计算"),
    ("scipy", "scipy", "滤波器 / 重采样"),
    ("transformers", "transformers", "HuBERT 特征提取"),
]
OPTIONAL_PACKAGES = [
    ("PySide6", "PySide6", "GUI（仅推理端需要）"),
    ("sounddevice", "sounddevice", "实时音频 IO（仅推理端需要）"),
]

# 音频扩展名单源在 rvc.core.constants（AUDIO_EXTS 全集 / NATIVE_EXTS 原生格式），
# 此处 re-export 供向导扫描使用（与 preprocess 指纹核对同一口径，新增格式只改 constants）。

EXIT_OK, EXIT_ENV, EXIT_RUNTIME = 0, 1, 2
BAR = "─" * 66

# 向导日志文件名。注意与实验目录内的 `train.log`（Trainer 写的逐 epoch 日志）区分：
# 本文件记录体检 / 参数 / 步骤进度。
LOG_NAME = "train_cli.log"

# 向导交互默认值（单源：能对上 dataclass 的项从 rvc.core.config 取，
# 切片时长从 rvc.core.constants 取 —— PreProcessor/Slicer 的签名默认值引用的同一常量）。
# 三层对外默认各自独立、取值一律不变，下面的具名常量只做「消除魔法数字」：
#   · epochs     —— 向导 250 / TrainConfig.epochs 2000
#   · save_every —— 向导 20  / TrainConfig.save_every_epoch 200
WIZARD_EPOCHS_DEFAULT = 250
WIZARD_SAVE_EVERY_DEFAULT = 20

# TrainConfig.exp_dir 为必填项，不可实例化，故直接读字段默认值
_TC_FIELD = TrainConfig.__dataclass_fields__
DEFAULTS = {
    "exp": "test",
    "sr": "48k",
    "epochs": WIZARD_EPOCHS_DEFAULT,
    "lr": _TC_FIELD["learning_rate"].default,
    "save_every": WIZARD_SAVE_EVERY_DEFAULT,
    "keep_ckpts": _TC_FIELD["keep_ckpts"].default,
    "per": SLICE_PERIOD_DEFAULT,
    "keep_models": _TC_FIELD["keep_models"].default,
    "hubert": HUBERT_DEFAULT,
    "use_pretrained": True,
}


def default_output_root() -> Path:
    """产物根目录默认值，按平台自适应。

    优先级：环境变量 RVC_OUT_ROOT > /root/autodl-tmp（存在即视为云实例）> 项目根。
    云上系统盘通常只有 30G，训练产物一律落数据盘；本地则落在项目根下
    （Windows 得到 <项目根>/logs/<实验名>）。
    """
    env = os.environ.get("RVC_OUT_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    cloud_root = Path("/root/autodl-tmp")
    if cloud_root.is_dir():
        return cloud_root
    return PROJECT_ROOT


def output_dirs(root: Path, exp_name: str) -> tuple[Path, Path]:
    """由产物根目录与实验名解析 (实验目录, 导出模型目录)。"""
    root = Path(root).expanduser()
    return root / "logs" / exp_name, root / "models" / exp_name


class Cancelled(Exception):
    """用户 Ctrl+C 取消（训练 CLI 专用）。

    ⚠️ 与 rvc.core.errors.CancelledError（推理侧，继承 BaseException）是两类
    不同的异常，勿混用/互捕：`except Exception` 能捕获本类但**漏掉**
    CancelledError——跨训练/推理两侧写代码时先确认异常的继承体系。
    """


class EnvFatal(Exception):
    """环境体检致命项缺失。"""
