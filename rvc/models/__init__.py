"""模型加载模块 — HuBERT, RMVPE, 缓存管理

hubert 子模块依赖 torch + transformers（重型），惰性导出；
inference_cache 轻量，启动即加载。
"""
from rvc.models.inference_cache import InferenceCache, default_inference_cache

__all__ = [
    "load_hubert",
    "InferenceCache",
    "default_inference_cache",
]


def __getattr__(name):
    if name == "load_hubert":
        from rvc.models.hubert import load_hubert
        return load_hubert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
