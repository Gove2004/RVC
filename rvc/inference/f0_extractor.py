"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口"""
import contextlib
import logging
import math
import sys
from abc import ABC, abstractmethod

import torch

from rvc.tools.cuda_graph import cuda_graph_enabled, run_cuda_graph

logger = logging.getLogger(__name__)

# F0 范围常量
F0_MIN = 50.0  # Hz - 人声最低基频
F0_MAX = 1100.0  # Hz - 人声最高基频

F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


class _FilteredStream:
    def __init__(self, stream, blocked_prefixes, blocked_contains):
        self.stream = stream
        self.blocked_prefixes = blocked_prefixes
        self.blocked_contains = blocked_contains

    def write(self, text):
        if not text or not text.strip():
            return len(text)
        stripped = text.lstrip()
        if any(stripped.startswith(prefix) for prefix in self.blocked_prefixes):
            return len(text)
        if any(token in text for token in self.blocked_contains):
            return len(text)
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def _suppress_third_party_output(*prefixes, contains=()):
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _FilteredStream(stdout, prefixes, contains) if stdout else stdout
    sys.stderr = _FilteredStream(stderr, prefixes, contains) if stderr else stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = stdout, stderr


def _normalize_f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
    """将连续 F0 归一化为离散 pitch 值 (1-255)。

    使用 Mel 频率标度进行归一化，将 F0 映射到 MIDI-like 的离散表示。

    Args:
        f0: 连续 F0 值 (Hz)

    Returns:
        pitch_coarse: 离散化的 pitch 值，范围 [1, 255]
    """
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return torch.round(f0_mel).long()


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

    def __init__(self, model_path: str, device: torch.device, is_half: bool) -> None:
        from rvc.models.rmvpe import RMVPE
        logger.info("加载 RMVPE")
        self.model = RMVPE(model_path, is_half=is_half, device=device)
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.model.infer_from_audio(audio, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)

        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        pitch_coarse = _normalize_f0_to_coarse(f0)

        return pitch_coarse, f0


class FCPEExtractor(F0Extractor):
    """FCPE F0 提取器"""

    def __init__(self, device: torch.device) -> None:
        from torchfcpe import spawn_bundled_infer_model
        logger.info("加载 FCPE")
        # 抑制 torchfcpe 的日志
        fcpe_logger = logging.getLogger("torchfcpe")
        saved_level = fcpe_logger.level
        fcpe_logger.setLevel(logging.ERROR)
        try:
            with _suppress_third_party_output(
                "[INFO]",
                "[INF0]",
                "[WARN]",
                ">",
                contains=("torchfcpe.mel_tools.nv_mel_extractor", "Librosa not found"),
            ):
                self.model = spawn_bundled_infer_model(device)
        finally:
            fcpe_logger.setLevel(saved_level)

        # CUDA Graph 需要 local_offsets 在 GPU 上（不能在 capture 期间 host→device copy）
        if cuda_graph_enabled(device) and hasattr(self.model, "model"):
            self.local_offsets = torch.arange(9, device=device, dtype=torch.long).view(1, 1, 9)
        else:
            self.local_offsets = None
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav_t = audio.to(self.device).unsqueeze(0).float()

        if cuda_graph_enabled(self.device):
            # CUDA Graph-safe FCPE infer: skip Wav2MelModule (has tensor-dependent conditionals),
            # capture only the stable-shape neural net + decoder.
            mel = self.model.wav2mel(wav_t, sr)

            def graphable_infer(mel_input):
                model = self.model.model
                latent = model(mel_input)
                batch, frames, _ = latent.shape
                cents = model.cent_table[None, None, :].expand(batch, frames, -1)

                confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
                local_index = self.local_offsets + (max_index - 4)
                local_index = local_index.clamp(0, model.out_dims - 1)
                local_cents = torch.gather(cents, -1, local_index)
                local_latent = torch.gather(latent, -1, local_index)
                decoded = torch.sum(local_cents * local_latent, dim=-1, keepdim=True) / torch.sum(local_latent, dim=-1, keepdim=True)

                confidence_mask = torch.ones_like(confidence)
                confidence_mask.masked_fill_(confidence <= 0.006, float("-inf"))
                decoded = decoded * confidence_mask
                return 10.0 * torch.pow(2.0, decoded / 1200.0)

            f0 = run_cuda_graph(
                self.model.model, "fcpe-core-local_argmax-0.006", graphable_infer, mel
            )
        else:
            f0 = self.model.infer(
                wav_t, sr=sr, decoder_mode="local_argmax", threshold=0.006,
            )

        f0 *= pow(2, f0_up_key / 12)

        # 转换为 Tensor
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()

        # Mel 归一化
        pitch_coarse = _normalize_f0_to_coarse(f0)

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
