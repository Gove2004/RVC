"""Synthesizer 模型加载器 — 纯权重加载（缓存由 ModelSessions 负责）。"""
import logging
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


@dataclass
class SynthesizerBundle:
    """加载结果的轻量载体（避免字典字符串键）。"""
    synthesizer: object
    model_sr: int
    use_f0: bool


class SynthesizerLoader:
    """Synthesizer 加载器 — 纯加载（缓存由 ModelSessions 统一负责）。"""

    def __init__(self, config):
        self.device = config.device
        self.is_half = config.is_half

    def load(self, pth_path):
        """加载 Synthesizer（无缓存，每次都是一次真实加载）。

        Args:
            pth_path: .pth 模型路径

        Returns:
            SynthesizerBundle
        """
        return self._load_pytorch(pth_path)

    def _load_pytorch(self, pth_path):
        """加载标准 PyTorch Synthesizer。"""
        # 安全前提：用户自备或官方导出的 .pth 模型文件，来源可信，故 weights_only=False。
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        model_sr = ckpt["config"][-1]
        use_f0 = bool(ckpt.get("f0", 1))

        from rvc.models.synthesizer import SynthesizerTrnMsNSFsid, SynthesizerTrnMsNSFsid_nono
        if use_f0:
            synthesizer = SynthesizerTrnMsNSFsid(*ckpt["config"], is_half=self.is_half)
        else:
            synthesizer = SynthesizerTrnMsNSFsid_nono(*ckpt["config"])

        # enc_q（PosteriorEncoder）仅训练时使用，推理 infer() 完全不调用。
        # 在 load_state_dict 之前删除，避免加载其权重（strict=False 会跳过缺失键），
        # 省 ~30MB 显存。不要改回 strict=True——那会因 enc_q 权重缺失直接报错。
        del synthesizer.enc_q

        result = synthesizer.load_state_dict(ckpt["weight"], strict=False)
        # strict=False 会静默吞掉一切键不匹配：拿错架构/错版本的 pth 加载后
        # 不报错、输出乱音极难排查。enc_q.* 是预期缺失，其余 missing/unexpected
        # 都值得告警。
        unexpected = [k for k in result.unexpected_keys if not k.startswith("enc_q.")]
        missing = [k for k in result.missing_keys if not k.startswith("enc_q.")]
        if unexpected:
            logger.warning("模型权重包含推理用不到的键（已忽略）: %s", unexpected[:8])
        if missing:
            logger.warning("模型权重缺失以下键（保持随机初始化，输出可能异常）: %s", missing[:8])
        del ckpt  # 释放 CPU 侧 checkpoint 副本（~200MB，fp32 权重字典）
        synthesizer.eval().to(self.device)
        if self.is_half:
            synthesizer.half()

        # CUDA Graph 已在 RuntimeDeviceConfig 初始化时探测，此处不重复

        return SynthesizerBundle(synthesizer, model_sr, use_f0)
