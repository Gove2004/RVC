"""模型加载模块 — HuBERT, RMVPE

hubert 子模块依赖 torch + transformers（重型），惰性导出。
"""

__all__ = [
    "load_hubert",
]


def __getattr__(name):
    if name == "load_hubert":
        from rvc.models.hubert import load_hubert
        return load_hubert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
