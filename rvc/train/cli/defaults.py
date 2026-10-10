"""训练向导存档 —— 把上次用过的参数作为下次运行的默认值。

首次运行（无存档）时各默认值来自内置 `DEFAULTS`、平台自适应的产物根目录，
以及 assets/configs/<sr>train_config.json（训练底参）。

向导问完全部问题后把本次答案原子写入 `TRAIN_CLI_DEFAULTS_FILE`，
下次运行直接以它们作为提问默认值。

存档是纯个人偏好，不进版本库；坏 JSON 会备份为 `.bak` 后回退内置默认值，
不静默吞掉用户数据（与 `gui/configs/config.py` 同一处理口径）。
"""
import json
import logging
import os
from pathlib import Path

from rvc.runtime.paths import TRAIN_CLI_DEFAULTS_FILE
from rvc.train.cli import DEFAULTS, default_output_root

logger = logging.getLogger(__name__)

# 参与存档的键（白名单）：新增提问项要一并加进来，删掉的项自动过期丢弃。
# 只存「用户填过的东西」—— device/fp16 是自动探测的。
PERSISTED_KEYS = (
    # ── 路径与规模 ──
    "input_dir", "output_root", "exp_name", "sr", "hubert",
    "batch_size", "lr", "epochs", "save_every", "keep_ckpts", "use_pretrained",
    # ── 预处理 / 切片 ──
    "per", "overlap", "highpass", "slice_threshold", "slice_min_length",
    "slice_min_interval", "slice_hop_size", "slice_max_sil_kept",
    # ── 训练底参（默认值来自 <sr>train_config.json）──
    "c_mel", "c_kl", "lr_decay", "seed", "segment_size", "betas", "eps",
    # ── 运行时 ──
    "num_workers", "rmvpe_threshold", "keep_models",
)

# 取值范围（下限, 上限，闭区间）；越界即丢弃、回退内置/json 默认。
# 与向导里 check 的口径保持一致，防止手改存档塞进跑不起来的数值。
_INT_RANGES = {
    "batch_size": (1, 4096),
    "epochs": (1, 10 ** 7),
    "save_every": (1, 10 ** 6),
    "keep_ckpts": (1, 10 ** 4),
    "keep_models": (0, 10 ** 4),
    "num_workers": (0, 256),
    "seed": (0, 2 ** 31 - 1),
    "segment_size": (1, 10 ** 7),
    "slice_min_length": (10, 60000),
    "slice_min_interval": (0, 60000),
    "slice_hop_size": (1, 1000),
    "slice_max_sil_kept": (0, 60000),
}
_FLOAT_RANGES = {
    "per": (1.0, 30.0),
    "overlap": (0.0, 30.0),
    "highpass": (10.0, 500.0),
    "slice_threshold": (-100.0, 0.0),
    "lr": (1e-9, 0.1),
    "c_mel": (0.0, 10000.0),
    "c_kl": (0.0, 1000.0),
    "lr_decay": (0.5, 1.0),
    "eps": (1e-15, 1.0),
    "rmvpe_threshold": (0.0, 1.0),
}


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_raw() -> dict:
    """读原始存档；缺失/损坏一律返回空 dict。"""
    path = TRAIN_CLI_DEFAULTS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            os.replace(path, backup)
            logger.warning("训练存档损坏，已备份到 %s：%s", backup, e)
        except Exception:
            logger.warning("训练存档损坏且备份失败：%s", e, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _sanitize(saved: dict) -> dict:
    """只放行合法项：换机器/手改文件后不会把非法值当成默认值。"""
    ok: dict = {}

    # 目录类：只沿用真实存在的路径。存档里的云盘/移动硬盘路径在当前机器上
    # 不存在时直接丢弃，回退到平台默认，免得默认值指向一个建不出来的盘。
    for key in ("input_dir", "output_root"):
        value = saved.get(key)
        if isinstance(value, str) and value.strip() and Path(value).expanduser().is_dir():
            ok[key] = value

    name = saved.get("exp_name")
    if isinstance(name, str) and name.strip():
        ok["exp_name"] = name.strip()

    if saved.get("sr") in ("40k", "48k"):
        ok["sr"] = saved["sr"]
    if saved.get("hubert") in ("base", "chinese"):
        ok["hubert"] = saved["hubert"]
    if isinstance(saved.get("use_pretrained"), bool):
        ok["use_pretrained"] = saved["use_pretrained"]

    for key, (low, high) in _INT_RANGES.items():
        value = saved.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
            ok[key] = value

    for key, (low, high) in _FLOAT_RANGES.items():
        value = saved.get(key)
        if _is_num(value) and low <= float(value) <= high:
            ok[key] = float(value)

    betas = saved.get("betas")
    if (isinstance(betas, (list, tuple)) and len(betas) == 2
            and all(_is_num(v) and 0 <= float(v) < 1 for v in betas)):
        ok["betas"] = [float(betas[0]), float(betas[1])]

    # 交叉校验：重叠必须小于切片时长，否则切片推进为零、会卡死
    if "overlap" in ok and "per" in ok and ok["overlap"] >= ok["per"]:
        ok.pop("overlap")
    return ok


def load_defaults() -> dict:
    """本次运行各项提问的默认值（存档优先，缺项/坏项回退内置与平台默认）。

    训练底参（c_mel / c_kl / lr_decay / seed / segment_size / betas / eps）缺省为
    None —— 表示「用当前采样率对应 json 里的值」，由向导在采样率确定后填入。
    """
    usable = {k: DEFAULTS[k] for k in (
        "sr", "hubert", "lr", "epochs", "save_every", "keep_ckpts", "keep_models",
        "use_pretrained", "per", "overlap", "highpass", "slice_threshold",
        "slice_min_length", "slice_min_interval", "slice_hop_size",
        "slice_max_sil_kept", "num_workers", "rmvpe_threshold",
    )}
    defaults = {
        "input_dir": "",
        "output_root": str(default_output_root()),
        "exp_name": DEFAULTS["exp"],
        "batch_size": None,  # None = 按显存现算（_suggest_batch）
        "c_mel": None,
        "c_kl": None,
        "lr_decay": None,
        "eps": None,
        "seed": None,
        "segment_size": None,  # None = 用当前采样率 json 的值
        "betas": None,
        **usable,
    }
    defaults.update(_sanitize(_load_raw()))
    return defaults


def save_defaults(cfg: dict) -> None:
    """把本次答案写为下次默认值（白名单键；失败只告警，不影响训练）。"""
    data = {key: cfg[key] for key in PERSISTED_KEYS if key in cfg}
    if "exp_name" in data:
        data["exp_name"] = str(data["exp_name"])
    if "betas" in data:
        data["betas"] = [float(v) for v in data["betas"]]
    path = TRAIN_CLI_DEFAULTS_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写：先写 .tmp 再 os.replace，写入中断不会损坏原存档
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("保存训练存档失败：%s", e)
