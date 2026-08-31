"""HuBERT 模型加载 — 使用 HuggingFace transformers 加载转换后的 HuBERT"""
import logging
import os

import torch
from torch import nn
from transformers import HubertModel
from transformers.utils.logging import disable_progress_bar

# 关闭 transformers 加载权重时的 tqdm 进度条（与统一日志格式冲突）
disable_progress_bar()

from rvc.models.inference_cache import default_inference_cache
from rvc.runtime.paths import HUBERT_ROOT

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


# HuBERT 特征器按子目录管理：assets/hubert/base/（原始 hubert_base）
# 与 assets/hubert/chinese/（腾讯 TencentGameMate/chinese-hubert-base，WenetSpeech 中文）。
# 训练与推理必须用同一特征器；每张模型卡牌/每次训练可独立选择 variant。
# HUBERT_ROOT 统一来自 rvc.runtime.paths（assets/hubert）
HUBERT_VARIANTS = ("base", "chinese")


def hubert_path(variant: str = "base") -> str:
    """返回指定 HuBERT 变体的模型目录（不校验存在性，由 load_hubert 统一报错）。"""
    return str(HUBERT_ROOT / variant)


def load_hubert(config, inference_cache=None, variant: str = "base"):
    inference_cache = inference_cache or default_inference_cache
    if variant not in HUBERT_VARIANTS:
        raise ValueError(f"未知 HuBERT 变体: {variant!r}（可选 {HUBERT_VARIANTS}）")
    model_path = hubert_path(variant)
    missing = [f for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin")
               if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FileNotFoundError(
            f"HuBERT 权重缺失: {model_path}/{missing[0]}（{variant} 特征器未就位，"
            f"请确认 assets/hubert/{variant}/ 三件套完整）"
        )

    cache_key = (config.device, config.is_half, variant)
    cached = inference_cache.get_hubert(cache_key)
    if cached is not None:
        logger.info("加载 HuBERT（缓存, %s）", variant)
        return cached

    dtype = torch.float16 if config.is_half else torch.float32
    logger.info("加载 HuBERT（transformers, %s, variant=%s）", dtype, variant)

    hubert_model = HubertModelWithFinalProj.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(config.device).eval()

    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()

    inference_cache.set_hubert(cache_key, hubert_model)
    return hubert_model
