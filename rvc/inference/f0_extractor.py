"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口"""
import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class F0Extractor(ABC):
    """F0 提取器抽象基类 — 统一接口。"""

    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        """提取 F0 (pitch)。

        Args:
            audio: 输入音频 (1D Tensor)
            sr: 采样率
            f0_up_key: 音高偏移（半音）

        Returns:
            (pitch_coarse, pitchf): 离散化 pitch 和连续 pitch
        """
        pass


class RMVPEExtractor(F0Extractor):
    """RMVPE F0 提取器"""

    def __init__(self, model_path: str, device: torch.device, is_half: bool):
        from rvc.models.rmvpe import RMVPE
        logger.info("加载 RMVPE")
        self.model = RMVPE(model_path, is_half=is_half, device=device)
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        # RMVPE 需要 numpy 输入
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        f0 = self.model.infer_from_audio(audio_np, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)

        # 转换为 Tensor
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()

        # Mel 归一化
        f0_mel_min = 1127 * torch.log(1 + torch.tensor(50.0) / 700)
        f0_mel_max = 1127 * torch.log(1 + torch.tensor(1100.0) / 700)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch_coarse = torch.round(f0_mel).long()

        return pitch_coarse, f0


class FCPEExtractor(F0Extractor):
    """FCPE F0 提取器"""

    def __init__(self, device: torch.device):
        from torchfcpe import spawn_bundled_infer_model
        logger.info("加载 FCPE")
        # 抑制 torchfcpe 的日志
        fcpe_logger = logging.getLogger("torchfcpe")
        saved_level = fcpe_logger.level
        fcpe_logger.setLevel(logging.ERROR)
        try:
            self.model = spawn_bundled_infer_model(device)
        finally:
            fcpe_logger.setLevel(saved_level)
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.model.infer(
            audio.to(self.device).unsqueeze(0).float(),
            sr=sr,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
        f0 *= pow(2, f0_up_key / 12)

        # 转换为 Tensor
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()

        # Mel 归一化
        f0_mel_min = 1127 * torch.log(1 + torch.tensor(50.0) / 700)
        f0_mel_max = 1127 * torch.log(1 + torch.tensor(1100.0) / 700)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch_coarse = torch.round(f0_mel).long()

        return pitch_coarse, f0


def create_f0_extractor(method: str, device: torch.device, is_half: bool, inference_cache) -> F0Extractor:
    """F0 提取器工厂函数 — 支持缓存。

    Args:
        method: "rmvpe" 或 "fcpe"
        device: 目标设备
        is_half: 是否使用半精度
        inference_cache: 推理缓存实例

    Returns:
        F0Extractor 实例
    """
    if method == "rmvpe":
        cache_key = (device, is_half)
        cached = inference_cache.get_rmvpe(cache_key)
        if cached is None:
            cached = RMVPEExtractor("assets/rmvpe/rmvpe.pt", device, is_half)
            inference_cache.set_rmvpe(cache_key, cached)
        return cached
    elif method == "fcpe":
        cache_key = device
        cached = inference_cache.get_fcpe(cache_key)
        if cached is None:
            cached = FCPEExtractor(device)
            inference_cache.set_fcpe(cache_key, cached)
        return cached
    else:
        raise ValueError(f"未知的 F0 提取方法: {method}")
