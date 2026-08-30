"""人声提纯 — 从 PyMSS 精简而来的本地人声提取/去混响/去和声/去杂音。

纯 PyTorch 实现（不需要 onnxruntime），CUDA 直跑，本地无时长限制。
权重需手动下载到 assets/separate/（链接见 registry.py 的 relpath，或用 model_urls(key) 取直链）。
重型依赖（torch / 模型代码）走惰性导入，避免拖慢 GUI 启动。
"""
from .registry import (
    AUDIO_EXTS,
    EXTRACT_KEYS,
    MODEL_DIR,
    MODELS,
    POST_KEYS,
    MS_BASE_URL,
    HF_BASE_URL,
    ModelSpec,
    get,
    missing,
)

__all__ = [
    "AUDIO_EXTS",
    "EXTRACT_KEYS",
    "HF_BASE_URL",
    "MODEL_DIR",
    "MODELS",
    "MS_BASE_URL",
    "POST_KEYS",
    "ModelSpec",
    "VocalSeparator",
    "get",
    "missing",
    "model_urls",
]


def model_urls(key: str) -> dict[str, str]:
    """返回该模型权重（及配置）在 ModelScope / HuggingFace 的直链，供手动下载。"""
    spec = MODELS[key]
    urls = {"ModelScope": f"{MS_BASE_URL}/{spec.relpath}", "HuggingFace": f"{HF_BASE_URL}/{spec.relpath}"}
    if spec.config_relpath:
        urls["配置 ModelScope"] = f"{MS_BASE_URL}/{spec.config_relpath}"
        urls["配置 HuggingFace"] = f"{HF_BASE_URL}/{spec.config_relpath}"
    return urls


def __getattr__(name):
    if name == "VocalSeparator":
        from .separator import VocalSeparator

        return VocalSeparator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
