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
    """Transformers HuBERT 加 RVC 需要的 final_proj 层。

    forward 直接可调用（transformers 约定，return_dict=True 默认返回
    BaseModelOutput），特征提取层用 outputs.last_hidden_state 取值，
    不再需要包装层。final_proj 是 RVC 导出模型的固定结构。
    """
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


HUBERT_MODEL_PATH = "assets/hubert"


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

    inference_cache.set_hubert(cache_key, hubert_model)
    return hubert_model
