"""推理模块 — VC 管线、运行时参数、F0 提取器、模型加载器

子模块（pipeline/f0_extractor/model_loader 等）依赖 torch，属重型模块，惰性导出；
params/offline_config 轻量。
"""
import importlib

__all__ = [
    "VCPipeline",
    "OfflineConfig",
    "Params",
    "F0Extractor",
    "RMVPEExtractor",
    "FCPEExtractor",
    "create_f0_extractor",
    "SynthesizerLoader",
]

_MODULE_MAP = {
    "VCPipeline": "rvc.inference.pipeline",
    "OfflineConfig": "rvc.inference.offline_config",
    "Params": "rvc.inference.params",
    "F0Extractor": "rvc.inference.f0_extractor",
    "RMVPEExtractor": "rvc.inference.f0_extractor",
    "FCPEExtractor": "rvc.inference.f0_extractor",
    "create_f0_extractor": "rvc.inference.f0_extractor",
    "SynthesizerLoader": "rvc.inference.model_loader",
}


def __getattr__(name):
    mod_name = _MODULE_MAP.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, name)
