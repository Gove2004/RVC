"""训练向导存档 —— 把上次用过的参数作为下次运行的默认值。

首次运行（无存档）时各默认值来自内置 `DEFAULTS` 与平台自适应的产物根目录；
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
# 只存「用户填过的东西」—— device/fp16 是自动探测的，per/keep_models 是固定值。
PERSISTED_KEYS = (
    "input_dir", "output_root", "exp_name", "sr", "hubert",
    "batch_size", "lr", "epochs", "save_every", "keep_ckpts", "use_pretrained",
)


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

    if isinstance(saved.get("batch_size"), int) and not isinstance(saved.get("batch_size"), bool):
        if saved["batch_size"] >= 1:
            ok["batch_size"] = saved["batch_size"]

    lr = saved.get("lr")
    if isinstance(lr, (int, float)) and not isinstance(lr, bool) and 0 < float(lr) <= 0.1:
        ok["lr"] = float(lr)

    for key in ("epochs", "save_every", "keep_ckpts"):
        value = saved.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            ok[key] = value

    if isinstance(saved.get("use_pretrained"), bool):
        ok["use_pretrained"] = saved["use_pretrained"]

    return ok


def load_defaults() -> dict:
    """本次运行各项提问的默认值（存档优先，缺项/坏项回退内置与平台默认）。"""
    defaults = {
        "input_dir": "",
        "output_root": str(default_output_root()),
        "exp_name": DEFAULTS["exp"],
        "sr": DEFAULTS["sr"],
        "hubert": DEFAULTS["hubert"],
        "batch_size": None,  # None = 按显存现算（_suggest_batch）
        "lr": DEFAULTS["lr"],
        "epochs": DEFAULTS["epochs"],
        "save_every": DEFAULTS["save_every"],
        "keep_ckpts": DEFAULTS["keep_ckpts"],
        "use_pretrained": DEFAULTS["use_pretrained"],
    }
    defaults.update(_sanitize(_load_raw()))
    return defaults


def save_defaults(cfg: dict) -> None:
    """把本次答案写为下次默认值（白名单键；失败只告警，不影响训练）。"""
    data = {key: cfg[key] for key in PERSISTED_KEYS if key in cfg}
    if "exp_name" in data:
        data["exp_name"] = str(data["exp_name"])
    path = TRAIN_CLI_DEFAULTS_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写：先写 .tmp 再 os.replace，写入中断不会损坏原存档
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("保存训练存档失败：%s", e)
