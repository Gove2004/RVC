"""HuBERT 模型加载 — 使用 HuggingFace transformers 加载转换后的 HuBERT"""
import logging

import torch
from torch import nn
from transformers import HubertModel
from transformers.utils.logging import disable_progress_bar

# 关闭 transformers 加载权重时的 tqdm 进度条（与统一日志格式冲突）
disable_progress_bar()

from rvc.models.inference_cache import default_inference_cache

logger = logging.getLogger(__name__)


class HubertModelWithFinalProj(HubertModel):
    """Transformers HuBERT 加 RVC 需要的 final_proj 层。"""
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


HUBERT_MODEL_PATH = "assets/hubert_base"


def load_hubert(config, inference_cache=None):
    inference_cache = inference_cache or default_inference_cache
    cache_key = (config.device, config.is_half)
    cached = inference_cache.get_hubert(cache_key)
    if cached is not None:
        logger.info("加载 HuBERT（缓存）")
        return cached

    dtype = torch.float16 if config.is_half else torch.float32
    logger.info("加载 HuBERT（transformers, %s）", dtype)

    hubert_model = HubertModelWithFinalProj.from_pretrained(
        HUBERT_MODEL_PATH,
        local_files_only=True,
    ).to(config.device).eval()

    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()

    wrapped = _HubertWrapper(hubert_model)
    inference_cache.set_hubert(cache_key, wrapped)
    return wrapped


class _HubertWrapper:
    """包装 transformers HuBERT，提供与 pipeline 兼容的 extract_features 接口。"""

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

    def __call__(self, input_values, attention_mask=None):
        """Callable forward — 供 feature_processing 直接调用。

        返回 tensor（last_hidden_state）。
        """
        kwargs = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "return_dict": True,
        }
        outputs = self.model(**kwargs)
        return outputs.last_hidden_state

