"""统一配置体系 — 训练/推理/GUI/持久化共用一套 dataclass。

历史上配置分散在 Params / InferGuiState / ModelConfig / RuntimeConfig /
EngineConfig / OfflineConfig 六个 dataclass，新增参数要改 5 处。
本模块是唯一配置源，所有层直接引用此处的 dataclass。

嵌套结构：
  AppConfig
  ├── inference: InferenceConfig   推理参数（音高/音色/保护/降噪/破音）
  ├── engine: EngineConfig         引擎参数（块长/设备/采样率模式）
  ├── active_model: str            当前激活模型路径
  └── models: list[ModelEntry]     模型列表
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# ── 子配置 ──────────────────────────────────────────────────────────

@dataclass
class BreakProtectConfig:
    """破音保护：高音超过临界 Hz 后按压缩比软收敛。

    ratio/knee 为内部固定默认值，用户只调 enable + src_hz。
    """
    enable: bool = True
    src_hz: float = 300.0       # 临界（源 Hz，extractor 内按 key 换算变声后）
    ratio: float = 0.4           # 高音区压缩比（内部固定）
    knee: float = 0.12            # 平滑膝宽比例（内部固定）


@dataclass
class DenoiseConfig:
    """输入侧谱减法降噪。"""
    enable: bool = False
    strength: float = 0.5


@dataclass
class InferenceConfig:
    """推理参数 — 实时/离线共用。"""
    pitch: int = 0                          # 音高偏移（半音）
    formant: float = 0.0                    # formant shift（原 gender 字段，[-2.5, +2.5]）
    protect: float = 0.5                    # 辅音保护强度 [0, 1]
    f0_method: str = "rmvpe"                # F0 提取器：rmvpe / fcpe
    rms_mix: float = 0.0                    # RMS 响度混合（0=目标响度，1=源响度）
    hubert_variant: str = "chinese"         # HuBERT 特征器：base / chinese（训练推理必须一致）
    break_protect: BreakProtectConfig = field(default_factory=BreakProtectConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class EngineConfig:
    """实时引擎参数 — 音频设备 + 延迟控制。"""
    block_time: float = 0.25                # 采样长度（秒）
    crossfade_time: float = 0.05            # 交叉淡化（秒）
    extra_time: float = 2.5                  # 额外上下文（秒）
    sr_mode: str = "model"                   # 采样率模式：model / device
    hostapi: str = ""                        # 音频 API 名称
    input_device: str = ""                   # 输入设备名称
    output_device: str = ""                  # 主输出设备名称
    output2_device: str = ""                 # 副输出设备名称（"" = 不使用）
    enable_out2: bool = False                # 是否启用副输出


@dataclass
class ModelEntry:
    """模型列表中的一条。"""
    name: str = ""
    pth: str = ""
    pitch: int = 0
    formant: float = 0.0
    hubert: str = "chinese"


@dataclass
class OfflineTask:
    """离线推理任务 = 推理参数 + 引擎参数 + 路径。"""
    input_path: str = ""
    output_path: str = ""
    model_path: str = ""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    pad_sec: float = 3.0                     # 首尾上下文 pad（秒）


@dataclass
class AppConfig:
    """应用完整配置 — GUI 持久化的根对象。"""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    active_model: str = ""
    models: list[ModelEntry] = field(default_factory=list)


# ── 序列化工具 ──────────────────────────────────────────────────────

def config_to_dict(cfg: Any) -> dict:
    """dataclass → 可 JSON 序列化的 dict（嵌套 dataclass 自动展开）。"""
    return asdict(cfg)


def dict_to_config(data: dict, cls: type) -> Any:
    """dict → dataclass（只取 cls 定义的字段，忽略多余键，缺省用默认值）。

    嵌套 dataclass 字段自动递归构造。
    """
    valid_names = {f.name for f in fields(cls)}
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        # 嵌套 dataclass：如果字段类型是 dataclass 且 val 是 dict，递归构造
        ftype = f.type
        if isinstance(ftype, str):
            # from __future__ import annotations 导致类型是字符串，这里不做递归
            # 嵌套字段由调用方显式构造
            pass
        kwargs[f.name] = val
    return cls(**kwargs)


def save_config_json(cfg: Any, path: str | Path) -> None:
    """原子写配置到 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(config_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_config_json(path: str | Path, cls: type = AppConfig) -> Any:
    """从 JSON 文件加载配置；文件不存在或损坏时返回默认配置。"""
    path = Path(path)
    if not path.exists():
        return cls()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return cls()
    return _from_dict(data, cls)


def _from_dict(data: dict, cls: type) -> Any:
    """递归构造嵌套 dataclass。"""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    valid = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, ftype in valid.items():
        if name not in data:
            continue
        val = data[name]
        # 解析字符串化的类型注解（from __future__ import annotations）
        actual = _resolve_type(ftype, cls)
        if actual is not None and dataclasses.is_dataclass(actual) and isinstance(val, dict):
            kwargs[name] = _from_dict(val, actual)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _resolve_type(ftype, cls):
    """把字符串化的类型注解解析为实际类型。"""
    if isinstance(ftype, type):
        return ftype
    if isinstance(ftype, str):
        # 从模块全局命名空间查找
        import sys
        mod = sys.modules.get(cls.__module__)
        if mod and hasattr(mod, ftype):
            return getattr(mod, ftype)
    return None
