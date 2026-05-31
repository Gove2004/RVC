"""HuBERT 模型加载 — 使用 fairseq 加载原始 HuBERT"""
import os
import logging

import torch

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
        """与 pipeline 兼容的接口。返回 tuple，[0] 是隐状态。"""
        with torch.no_grad():
            inputs = {
                "source": source,
                "padding_mask": padding_mask,
                "output_layer": output_layer,
            }
            logits = self.model.extract_features(**inputs)
        return (logits[0],)


def load_hubert(config):
    """使用 fairseq 加载 HuBERT 模型。"""
    hubert_path = "assets/hubert/hubert_base.pt"

    if not os.path.exists(hubert_path):
        old_path = "D:/RVC20240604Nvidia50x0/assets/hubert/hubert_base.pt"
        if os.path.exists(old_path):
            import shutil
            os.makedirs(os.path.dirname(hubert_path), exist_ok=True)
            shutil.copy2(old_path, hubert_path)
            logger.info("已从 %s 复制 HuBERT 模型", old_path)
        else:
            raise FileNotFoundError(f"找不到 HuBERT 模型: {hubert_path}")

    logger.info("使用 fairseq 加载 HuBERT: %s", hubert_path)

    # fairseq 的 torch.load 需要 weights_only=False
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    try:
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], suffix=""
        )
        hubert_model = models[0]
    finally:
        torch.load = _original_load

    hubert_model = HubertWrapper(hubert_model)
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    logger.info("HuBERT 模型加载完成 (fairseq)")
    return hubert_model.eval()
