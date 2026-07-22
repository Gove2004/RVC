"""Synthesizer 模型加载器 — PyTorch 权重加载 + 缓存"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


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
            dict: {"net_g": model, "tgt_sr": int, "if_f0": int, "version": str}
        """
        cached = self.inference_cache.get_synthesizer(pth_path)
        if cached:
            logger.info("使用缓存 Synthesizer: %s", os.path.basename(pth_path))
            return cached

        logger.info("加载 Synthesizer")
        result = self._load_pytorch(pth_path)
        self.inference_cache.set_synthesizer(pth_path, result)
        return result

    def _load_pytorch(self, pth_path):
        """加载标准 PyTorch Synthesizer。"""
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        tgt_sr = ckpt["config"][-1]
        if_f0 = ckpt.get("f0", 1)
        version = ckpt.get("version", "v2")
        n_spk = ckpt["config"][-3] = ckpt["weight"]["emb_g.weight"].shape[0]

        from rvc.synthesizer import SynthesizerTrnMsNSFsid, SynthesizerTrnMsNSFsid_nono
        if if_f0 == 1:
            net_g = SynthesizerTrnMsNSFsid(*ckpt["config"], is_half=self.is_half)
        else:
            net_g = SynthesizerTrnMsNSFsid_nono(*ckpt["config"])

        net_g.load_state_dict(ckpt["weight"], strict=False)
        net_g.eval().to(self.device)
        if self.is_half:
            net_g.half()

        return {
            "net_g": net_g,
            "tgt_sr": tgt_sr,
            "if_f0": if_f0,
            "version": version,
        }
