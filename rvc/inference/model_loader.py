"""Synthesizer 模型加载器 — 支持 PyTorch + JIT 加速 + 缓存"""
import logging
import os
from io import BytesIO

import torch

logger = logging.getLogger(__name__)


class SynthesizerLoader:
    """Synthesizer 加载器 — 封装加载、JIT 导出和缓存逻辑。"""

    def __init__(self, config, inference_cache):
        self.config = config
        self.device = config.device
        self.is_half = config.is_half
        self.inference_cache = inference_cache

    def load(self, pth_path):
        """加载 Synthesizer（支持 JIT 加速和缓存）。

        Args:
            pth_path: .pth 模型路径

        Returns:
            dict: {"net_g": model, "tgt_sr": int, "if_f0": int, "version": str}
        """
        # 检查缓存
        cached = self.inference_cache.get_synthesizer(pth_path)
        if cached:
            logger.info("使用缓存 Synthesizer: %s", os.path.basename(pth_path))
            return cached

        # 尝试加载 JIT 模型
        if self.config.use_jit:
            jit_result = self._try_load_jit(pth_path)
            if jit_result:
                self.inference_cache.set_synthesizer(pth_path, jit_result)
                return jit_result

        # 降级到标准 PyTorch 模型
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

    def _try_load_jit(self, pth_path):
        """尝试加载 JIT 模型（如果存在且设备匹配）。

        Returns:
            dict or None: JIT 模型信息，或 None（需要降级）
        """
        jit_path = pth_path.rstrip(".pth")
        jit_path += ".half.jit" if self.is_half else ".jit"

        if not os.path.exists(jit_path):
            logger.info("导出 JIT: %s", os.path.basename(jit_path))
            return self._export_jit(pth_path, jit_path)

        try:
            from rvc.models import jit as jit_module
            cpt = jit_module.load(jit_path)
            model_device = cpt.get("device", "cpu")

            # 设备不匹配，重新导出
            if model_device != str(self.device):
                logger.info("重新导出 JIT（设备: %s → %s）", model_device, self.device)
                return self._export_jit(pth_path, jit_path)

            # 加载 JIT 模型
            net_g = torch.jit.load(BytesIO(cpt["model"]), map_location=self.device)
            net_g.infer = net_g.forward
            net_g.eval().to(self.device)

            logger.info("加载 JIT: %s", os.path.basename(jit_path))
            return {
                "net_g": net_g,
                "tgt_sr": cpt["config"][-1],
                "if_f0": cpt.get("f0", 1),
                "version": cpt.get("version", "v2"),
            }

        except Exception as e:
            logger.warning("JIT 加载失败，降级到 PyTorch: %s", e)
            return None

    def _export_jit(self, pth_path, jit_path):
        """导出 JIT 模型。

        Returns:
            dict or None: 导出成功返回模型信息，失败返回 None
        """
        try:
            from rvc.models import jit as jit_module
            logger.info("导出 JIT: %s", os.path.basename(jit_path))

            cpt = jit_module.synthesizer_jit_export(
                pth_path,
                mode="script",
                save_path=jit_path,
                device=self.device,
                is_half=self.is_half,
            )

            # 直接加载刚导出的 JIT 模型
            net_g = torch.jit.load(BytesIO(cpt["model"]), map_location=self.device)
            net_g.infer = net_g.forward
            net_g.eval().to(self.device)

            logger.info("导出 JIT 成功")
            return {
                "net_g": net_g,
                "tgt_sr": cpt["config"][-1],
                "if_f0": cpt.get("f0", 1),
                "version": cpt.get("version", "v2"),
            }

        except Exception as e:
            logger.warning("JIT 导出失败，降级到 PyTorch: %s", e, exc_info=True)
            return None
