"""Synthesizer 模型加载器 — PyTorch 权重加载 + 缓存"""
import logging
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


@dataclass
class SynthesizerBundle:
    """加载结果的轻量载体（避免字典字符串键）。"""
    synthesizer: object
    target_sr: int
    use_f0: int


class SynthesizerLoader:
    """Synthesizer 加载器 — 封装加载和缓存逻辑。"""

    def __init__(self, config, inference_cache):
        self.config = config
        self.device = config.device
        self.is_half = config.is_half
        self.inference_cache = inference_cache

    def load(self, pth_path):
        """加载 Synthesizer。

        Args:
            pth_path: .pth 模型路径

        Returns:
            SynthesizerBundle
        """
        cached = self.inference_cache.get_synthesizer(pth_path)
        if cached:
            logger.info("  · Synthesizer（缓存）")
            return cached

        logger.info("  · Synthesizer")
        result = self._load_pytorch(pth_path)
        self.inference_cache.set_synthesizer(pth_path, result)
        return result

    def _load_pytorch(self, pth_path):
        """加载标准 PyTorch Synthesizer。"""
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        target_sr = ckpt["config"][-1]
        use_f0 = ckpt.get("f0", 1)

        from rvc.synthesizer import SynthesizerTrnMsNSFsid, SynthesizerTrnMsNSFsid_nono
        if use_f0 == 1:
            synthesizer = SynthesizerTrnMsNSFsid(*ckpt["config"], is_half=self.is_half)
        else:
            synthesizer = SynthesizerTrnMsNSFsid_nono(*ckpt["config"])

        synthesizer.load_state_dict(ckpt["weight"], strict=False)
        synthesizer.eval().to(self.device)
        if self.is_half:
            synthesizer.half()

        # CUDA Graph 已在 Config 初始化时探测，此处不再重复

        return SynthesizerBundle(synthesizer, target_sr, use_f0)
