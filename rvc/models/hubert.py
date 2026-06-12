"""HuBERT 模型加载 — 使用 fairseq 加载原始 HuBERT"""
import os
import logging
import warnings

import torch

from rvc.models.cache import default_inference_cache

logger = logging.getLogger(__name__)


class HubertWrapper:
    """包装 fairseq HuBERT，提供与 pipeline 兼容的 extract_features 接口"""

    def __init__(self, model):
        self.model = model

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def half(self):
        self.model = self.model.half()
        return self

    def float(self):
        self.model = self.model.float()
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def extract_features(self, source, padding_mask=None, output_layer=12):
        with torch.no_grad():
            inputs = {
                "source": source,
                "padding_mask": padding_mask,
                "output_layer": output_layer,
            }
            logits = self.model.extract_features(**inputs)
        return (logits[0],)


def load_hubert(config, inference_cache=None):
    inference_cache = inference_cache or default_inference_cache
    cache_key = (config.device, config.is_half)
    cached = inference_cache.get_hubert(cache_key)
    if cached is not None:
        logger.info("加载 HuBERT（缓存）")
        return cached

    hubert_path = "assets/hubert/hubert_base.pt"
    if not os.path.exists(hubert_path):
        raise FileNotFoundError(f"找不到 HuBERT 模型: {hubert_path}")

    logger.info("加载 HuBERT")
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    _quiet_loggers = ["fairseq.tasks.hubert_pretraining", "fairseq.models.hubert.hubert",
                      "fairseq.models.hubert", "fairseq"]
    _saved_levels = {}
    for name in _quiet_loggers:
        l = logging.getLogger(name)
        _saved_levels[name] = l.level
        l.setLevel(logging.WARNING)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
            from fairseq import checkpoint_utils
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([hubert_path], suffix="")
        hubert_model = models[0]
    finally:
        torch.load = _original_load
        for name, level in _saved_levels.items():
            logging.getLogger(name).setLevel(level)

    hubert_model = HubertWrapper(hubert_model)
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    model = hubert_model.eval()
    inference_cache.set_hubert(cache_key, model)
    return model

