"""GUI 状态持久化配置。

真实配置的唯一权威 = assets/configs/save_state.json（rvc.runtime.paths.STATE_FILE
定位）。读取出的 dict 由 gui.state.bindings.params_from_dict 逐字段填入
InferenceParams（InferenceParams 没有 load() 方法，不接受整对象反序列化）。
"""
import logging
import threading

from rvc.io.atomic import atomic_write_text, load_json_or_default
from rvc.runtime.paths import STATE_FILE

logger = logging.getLogger(__name__)
_lock = threading.Lock()

_DEFAULT_STATE = {"gui": {}}


def load_config() -> dict:
    return load_json_or_default(STATE_FILE, dict(_DEFAULT_STATE), logger=logger)


def save_config(state: dict):
    with _lock:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            STATE_FILE,
            json.dumps(state, ensure_ascii=False, indent=2),
        )
