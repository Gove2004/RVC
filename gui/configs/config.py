"""GUI 状态持久化配置。"""
import json
import logging
import os
import threading

from rvc.runtime import config_path

logger = logging.getLogger(__name__)
_lock = threading.Lock()

_DEFAULT_STATE = {"gui": {}, "train": {}, "models": []}


def load_config() -> dict:
    state_file = config_path()
    if not state_file.exists():
        return dict(_DEFAULT_STATE)
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as e:
        # 坏 JSON 不静默吞：备份原文件（避免丢失用户数据），再返回默认。
        # 直接 return {} 会让模型列表/全部设置静默丢失，且无法恢复。
        backup = state_file.with_suffix(state_file.suffix + ".bak")
        try:
            os.replace(state_file, backup)
            logger.warning("配置文件损坏，已备份到 %s：%s", backup, e)
        except Exception:
            logger.warning("配置文件损坏且备份失败：%s", e, exc_info=True)
        return dict(_DEFAULT_STATE)


def save_config(data: dict):
    with _lock:
        state_file = config_path()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        # 原子写：先写临时文件，再 os.replace 替换。写入中断（崩溃/断电）
        # 不会损坏原文件（replace 是同文件系统原子操作）。
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, state_file)
