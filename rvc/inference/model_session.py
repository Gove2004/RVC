"""推理模型会话管理 — 统一管理 HuBERT/F0/Synthesizer 的加载与缓存。

ModelSessionManager 是模型生命周期的唯一入口：
- load(): 加载模型（HuBERT + Synthesizer），返回 ModelSession
- get_f0_extractor(): 获取 F0 提取器（RMVPE/FCPE，带缓存）
- clear_all(): 清除所有模型缓存和 CUDA Graph（停止/切换模型时调用）

realtime_engine 和 offline_manager 都通过本管理器获取模型，
避免缓存清除逻辑分散在多处导致遗漏（如之前的 FCPE CUDA Graph 未清除 bug）。
"""
import logging
import os
from dataclasses import dataclass

from rvc.core.errors import ModelLoadError
from rvc.inference.model_loader import SynthesizerLoader
from rvc.models.hubert import load_hubert
from rvc.inference.cuda_graph import clear_cuda_graph_cache

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    """一个模型文件的完整推理会话（HuBERT/Synthesizer 均为共享缓存实例）。"""
    hubert: object
    synthesizer: object
    target_sr: int
    use_f0: int


class ModelSessionManager:
    """模型会话管理器 — 统一管理模型加载、缓存与 CUDA Graph 清除。"""

    def __init__(self, device_config, inference_cache=None):
        """
        Args:
            device_config: 含 device / is_half 的配置对象
            inference_cache: InferenceCache 实例（None 则用默认全局缓存）
        """
        from rvc.inference.inference_cache import default_inference_cache
        self.device = device_config.device
        self.is_half = device_config.is_half
        self.inference_cache = inference_cache or default_inference_cache
        self._current_pth = None
        self._current_session = None

    def load(self, pth_path: str, hubert_variant: str = "chinese") -> ModelSession:
        """加载模型（HuBERT + Synthesizer），返回 ModelSession。

        同一 pth 重复加载时直接返回缓存的 session。
        """
        if self._current_pth == pth_path and self._current_session is not None:
            return self._current_session

        logger.info("加载 %s", os.path.basename(pth_path))
        try:
            hubert = load_hubert(
                _DeviceConfig(self.device, self.is_half),
                self.inference_cache,
                variant=hubert_variant,
            )
            loader = SynthesizerLoader(_DeviceConfig(self.device, self.is_half), self.inference_cache)
            syn = loader.load(pth_path)
            synthesizer = syn.synthesizer
        except Exception as e:
            raise ModelLoadError(f"模型加载失败 [{os.path.basename(pth_path)}]: {e}") from e

        # 所有 Synthesizer 变体都继承 remove_weight_norm；保留异常兜底（权重可能已是移除状态）
        try:
            synthesizer.remove_weight_norm()
        except Exception as e:
            logger.warning("移除 weight_norm 失败：%s", e)

        # 清除旧模型的 CUDA Graph 缓存（避免形状/设备不匹配）
        clear_cuda_graph_cache(synthesizer)
        clear_cuda_graph_cache(hubert)

        session = ModelSession(
            hubert=hubert,
            synthesizer=synthesizer,
            target_sr=syn.target_sr,
            use_f0=syn.use_f0,
        )
        self._current_pth = pth_path
        self._current_session = session
        return session

    def get_f0_extractor(self, method: str):
        """获取 F0 提取器（RMVPE/FCPE，带缓存）。

        Args:
            method: "rmvpe" 或 "fcpe"
        """
        from rvc.inference.f0_extractor import create_f0_extractor
        # model_session 不持有 InferenceConfig，暂时传 None（回退到 global experimental_config）
        # 后续可以把 config 传到 ModelSessionManager
        return create_f0_extractor(method, self.device, self.is_half, self.inference_cache, config=None)

    def clear_all(self) -> None:
        """清除所有模型缓存和 CUDA Graph（停止/切换模型时调用）。

        清除范围：
        - F0 提取器（RMVPE/FCPE）的 CUDA Graph
        - 当前 session 的 synthesizer/hubert 的 CUDA Graph
        - 推理缓存中的 synthesizer 缓存
        """
        # F0 提取器 CUDA Graph
        self.inference_cache.clear_f0_cuda_graph_caches()

        # 当前 session 的 CUDA Graph
        if self._current_session is not None:
            clear_cuda_graph_cache(self._current_session.synthesizer)
            clear_cuda_graph_cache(self._current_session.hubert)

        # 清除 synthesizer 缓存（LRU 中的旧模型）
        for syn_bundle in self.inference_cache._synthesizer.values():
            if hasattr(syn_bundle, 'synthesizer'):
                clear_cuda_graph_cache(syn_bundle.synthesizer)

        self._current_pth = None
        self._current_session = None

    @property
    def current_session(self) -> ModelSession | None:
        """当前加载的模型会话（未加载时为 None）。"""
        return self._current_session


class _DeviceConfig:
    """轻量级设备配置适配器，供 load_hubert / SynthesizerLoader 使用。"""
    def __init__(self, device: str, is_half: bool):
        self.device = device
        self.is_half = is_half
