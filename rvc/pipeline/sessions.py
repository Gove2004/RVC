"""推理模型会话管理 — 统一管理 HuBERT/Synthesizer 的加载与缓存。

ModelSessions 是模型生命周期的唯一入口：
- load(): 查缓存 → 未命中调纯加载器 → 回填缓存，返回 ModelSession

缓存策略单源在本层（两个模型同一套流程），加载器（load_hubert /
SynthesizerLoader）保持无状态、只负责「把文件变成对象」——避免出现
「一个模型在加载器里缓存、另一个在会话层缓存」的分裂。

F0 提取器（RMVPE/FCPE）不在此层管理：提取器带阈值等运行时参数，
由 pipeline 在推理时经 create_f0_extractor 工厂直接创建/复用。
"""
import gc
import logging
import os
from dataclasses import dataclass

import torch

from rvc.core.errors import ModelLoadError
from rvc.models.hubert import load_hubert
from rvc.pipeline.cache import default_inference_cache
from rvc.pipeline.loader import SynthesizerLoader
from rvc.runtime.graph import purge_cuda_graphs

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    """一个模型文件的完整推理会话（HuBERT/Synthesizer 均为共享缓存实例）。"""
    hubert: object
    synthesizer: object
    target_sr: int
    use_f0: bool


class ModelSessions:
    """模型会话层 — 统一管理模型加载与缓存。"""

    def __init__(self, device: str, is_half: bool, inference_cache=None):
        """
        Args:
            device: 目标设备（如 "cuda:0"）
            is_half: 是否 FP16
            inference_cache: InferenceCache 实例（None 则用默认全局缓存）
        """
        self.device = device
        self.is_half = is_half
        self.inference_cache = inference_cache or default_inference_cache

    def _device_config(self) -> "_DeviceConfig":
        """供 load_hubert / SynthesizerLoader 使用的轻量设备配置适配器。"""
        return _DeviceConfig(self.device, self.is_half)

    def load(self, pth_path: str, hubert_variant: str = "chinese") -> ModelSession:
        """加载模型（HuBERT + Synthesizer），返回 ModelSession。

        两个模型都先查 InferenceCache：命中直接复用（切回旧模型秒开），
        未命中才调加载器并回填。
        """
        logger.info("加载 %s", os.path.basename(pth_path))
        # 加载前先回收 CUDA 缓存碎片，确保最大可用显存（避免碎片导致 OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            # HuBERT 的 key 含变体：base 与 chinese 是两套不同权重的特征器
            hubert_key = (self.device, self.is_half, hubert_variant)
            hubert = self.inference_cache.get_hubert(hubert_key)
            if hubert is None:
                hubert = load_hubert(self._device_config(), variant=hubert_variant)
                self.inference_cache.set_hubert(hubert_key, hubert)
            else:
                logger.info("  · HuBERT（缓存, %s）", hubert_variant)

            # Synthesizer 的 key 必须含设备与精度：权重是按加载时的 dtype/device
            # 落地的，只按路径缓存会在设备/精度切换后复用错误副本。
            syn_key = (self.device, self.is_half, pth_path)
            syn = self.inference_cache.get_synthesizer(syn_key)
            if syn is None:
                syn = SynthesizerLoader(self._device_config()).load(pth_path)
                self.inference_cache.set_synthesizer(syn_key, syn)
            else:
                logger.info("  · Synthesizer（缓存）")
            synthesizer = syn.synthesizer
        except Exception as e:
            raise ModelLoadError(f"模型加载失败 [{os.path.basename(pth_path)}]: {e}") from e

        # 所有 Synthesizer 变体都继承 remove_weight_norm；保留异常兜底（权重可能已是移除状态）
        try:
            synthesizer.remove_weight_norm()
        except Exception as e:
            logger.warning("移除 weight_norm 失败：%s", e)

        # 清除旧模型的 CUDA Graph 缓存（避免形状/设备不匹配）
        purge_cuda_graphs(synthesizer, hubert)

        # 模型全部加载到 GPU 后，强制回收 CPU 侧残留的 checkpoint 副本
        # （torch.load(map_location="cpu") 的 ckpt 已 del，但 Python 分配器可能
        # 未立即归还 OS；gc.collect() 触发引用计数清理，省 ~500-700MB 系统内存）
        gc.collect()

        return ModelSession(
            hubert=hubert,
            synthesizer=synthesizer,
            target_sr=syn.target_sr,
            use_f0=syn.use_f0,
        )


class _DeviceConfig:
    """轻量级设备配置适配器，供 load_hubert / SynthesizerLoader 使用。"""
    def __init__(self, device: str, is_half: bool):
        self.device = device
        self.is_half = is_half
