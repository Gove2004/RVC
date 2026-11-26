"""HuBERT 模型加载 — 纯 torch 复刻，推理/训练均不依赖 transformers。

复刻体在 rvc.models.hubert_replica（模块命名对齐 transformers，权重直装，
三变体 fp16/fp32 实测 max|Δ|=0）。为什么复刻：`from transformers import HubertModel`
会连带拉起 torch._dynamo + sklearn，导入耗时数秒（热）乃至 70s（冷启杀软首扫）并常驻
约 160MB 私有内存，而 HuBERT 在本项目只作冻结特征器，用不到那套 machinery。
详见 hubert_replica 模块文档。

本模块只保留加载入口与变体校验；设备/精度由调用方经 config 传入。
"""
import logging
import os

import torch

from rvc.core.config import HUBERT_DEFAULT, HUBERT_VARIANTS
from rvc.core.errors import FeatureExtractionError
from rvc.models.hubert_replica import load_replica
from rvc.runtime.paths import HUBERT_ROOT

logger = logging.getLogger(__name__)

# HuBERT 特征器按子目录管理：assets/hubert/<variant>/（base = RVC 生态标准的
# hubert_base；chinese = 腾讯 chinese-hubert-base；ktum = KtUltraMorph 自训练版）。
# 训练与推理必须用同一特征器；变体清单与缺省值单源在 rvc.core.config。
__all__ = ["HUBERT_VARIANTS", "HUBERT_DEFAULT", "hubert_path", "load_hubert"]


def hubert_path(variant: str = HUBERT_DEFAULT) -> str:
    """返回指定 HuBERT 变体的模型目录（不校验存在性，由 load_hubert 统一报错）。"""
    return str(HUBERT_ROOT / variant)


def load_hubert(config, variant: str = HUBERT_DEFAULT):
    """加载 HuBERT 特征器（纯加载，不做缓存）。

    缓存策略属于调用方（推理 InferenceCache / 训练侧 _hubert_cache），
    模型层只负责把权重装到设备上。config 需带 device / is_half。
    """
    if variant not in HUBERT_VARIANTS:
        raise FeatureExtractionError(f"未知 HuBERT 变体: {variant!r}（可选 {HUBERT_VARIANTS}）")
    model_path = hubert_path(variant)
    # 复刻只需 config.json + pytorch_model.bin（preprocessor_config.json 是
    # transformers 音频预处理用，复刻不经它，故不再校验）
    missing = [f for f in ("config.json", "pytorch_model.bin")
               if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FeatureExtractionError(
            f"HuBERT 权重缺失: {model_path}/{missing[0]}（{variant} 特征器未就位，"
            f"请确认 assets/hubert/{variant}/ 含 config.json 与 pytorch_model.bin）"
        )

    dtype = torch.float16 if config.is_half else torch.float32
    logger.info("  · HuBERT（%s, variant=%s）", dtype, variant)
    return load_replica(model_path, config.device, config.is_half)
