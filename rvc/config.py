"""统一配置体系 — 训练/推理/GUI/持久化共用一套 dataclass。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

HUBERT_DEFAULT = "chinese"


@dataclass
class BreakProtectConfig:
    enable: bool = True
    src_hz: float = 300.0
    ratio: float = 0.4
    knee: float = 0.12


@dataclass
class DenoiseConfig:
    enable: bool = False
    strength: float = 0.5


@dataclass
class InferenceConfig:
    pitch: int = 0
    formant: float = 0.0
    protect: float = 0.5
    f0_method: str = "rmvpe"
    rms_mix: float = 0.0
    hubert_variant: str = "chinese"
    break_protect: BreakProtectConfig = field(default_factory=BreakProtectConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class EngineConfig:
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    sr_mode: str = "model"
    hostapi: str = ""
    input_device: str = ""
    output_device: str = ""
    output2_device: str = ""
    enable_out2: bool = False


@dataclass
class ModelEntry:
    name: str = ""
    pth: str = ""
    pitch: int = 0
    formant: float = 0.0
    hubert: str = "chinese"


@dataclass
class OfflineTask:
    input_path: str = ""
    output_path: str = ""
    model_path: str = ""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    pad_sec: float = 3.0


@dataclass
class AppConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    active_model: str = ""
    models: list[ModelEntry] = field(default_factory=list)


def config_to_dict(cfg: Any) -> dict:
    return asdict(cfg)


def dict_to_config(data: dict, cls: type) -> Any:
    return cls(**data)


def save_config_json(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(config_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_config_json(path: str | Path, cls: type = AppConfig) -> Any:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _from_dict(data, cls)


def _from_dict(data: dict, cls: type) -> Any:
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        actual = _resolve_type(f.type, cls)
        if actual is not None and dataclasses.is_dataclass(actual) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(val, actual)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _resolve_type(ftype, cls):
    if isinstance(ftype, type):
        return ftype
    if isinstance(ftype, str):
        import sys
        mod = sys.modules.get(cls.__module__)
        if mod and hasattr(mod, ftype):
            return getattr(mod, ftype)
    return None
