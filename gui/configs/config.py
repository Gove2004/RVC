"""GUI 状态持久化配置。"""
import json

from rvc.runtime import config_path


def load_config() -> dict:
    state_file = config_path()
    if not state_file.exists():
        return {"gui": {}, "train": {}, "models": []}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"gui": {}, "train": {}, "models": []}


def save_config(data: dict):
    state_file = config_path()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
