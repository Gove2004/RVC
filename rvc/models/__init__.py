"""模型加载模块 — HuBERT（惰性导出）。

hubert 子模块经 __getattr__ 惰性加载；HuBERT 用纯 torch 复刻
（rvc.models.hubert_replica），不依赖 transformers。RMVPE 经 rvc.models.rmvpe
子模块直导，不经本包门面。
"""

__all__ = [
    "load_hubert",
]


def __getattr__(name: str) -> object:
    if name == "load_hubert":
        from rvc.models.hubert import load_hubert
        return load_hubert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
