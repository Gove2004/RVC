"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口"""
import contextlib
import logging
import math
import sys
from abc import ABC, abstractmethod

import torch

from rvc.runtime.paths import RMVPE_PATH
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


# 破音保护（用户核心瑕疵：高音破音/沙哑）。作用在【变声后】音高上：
# 变声后 ≤ 临界 完全原样（说话/唱歌动态 100% 保留），> 临界 指数软收敛——
# 保留不同高音之间的相对起伏（动态音调），同时渐近收敛不冲爆。
# 用户直觉参照是源赫兹（"我唱到 300Hz 就破"），GUI 填源值，内部换算变声后。
BREAK_PROTECT_DEFAULT_SRC_HZ = 300.0  # 破音临界（源 Hz）：超过开始收敛
BREAK_PROTECT_OVER = 0.25             # 渐近上限 = 临界×(1+OVER)（超临界保留 25% 动态区）


def apply_f0_break_protect(f0: torch.Tensor, critical_hz: float) -> torch.Tensor:
    """破音保护（软收敛/动态音调）：≤临界完全原样，>临界渐近收敛不抹平。

    作用在【变声后】音高上（pitch 平移之后）——用户方案：正常说话 115、
    破音临界 300（源）→ 变声 +12 后正常 230、破音 600。变声后 ≤600 的
    部分完全原样（说话/唱歌动态 100% 保留）。

    >600 的部分：指数软收敛 `y = U - K·exp(-(f0-C)/K)`，U=C×1.25 为渐近
    上限、K=U-C。f0=C 处 y=C 且斜率 =1（与左侧原样段无缝衔接），之后
    斜率平滑降到 0——**不同高音仍拉开差距（720→683、800→710），不会像
    硬切那样全锁成一个音（唱歌高音一个调）**，同时输出有界（永不超
    C×1.25），模型收到的音高远低于原始冲爆值，不再沙哑。

    单调 + 连续（一阶导数也连续）+ 有界 + 保留动态，四者全部满足。

    Args:
        f0: 变声后的连续 F0 值 (Hz, GPU tensor)
        critical_hz: 破音临界（变声后 Hz）
    """
    if critical_hz <= 0 or f0 is None:
        return f0
    U = critical_hz * (1.0 + BREAK_PROTECT_OVER)
    K = U - critical_hz
    x = (f0 - critical_hz) / K          # >0 的帧才收敛（mask 过滤）
    y_hi = U - K * torch.exp(-x)
    return torch.where(f0 > critical_hz, y_hi, f0)


class F0Extractor(ABC):
    """F0 提取器抽象基类 — 统一接口。"""

    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """提取 F0 (pitch)。

        Args:
            audio: 输入音频 (1D Tensor)
            sr: 采样率
            f0_up_key: 音高偏移（半音）
            f0_proc: (破音保护开关, 破音临界[源Hz]) 或 None

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

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.model.infer_from_audio(audio, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)

        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        # 破音保护（变声后域）：f0_proc=(开关, 破音临界[源Hz])，内部换算变声后
        if f0_proc and f0_proc[0]:
            f0 = apply_f0_break_protect(f0, f0_proc[1] * pow(2, f0_up_key / 12))
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

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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

        # 破音保护（变声后域）：f0_proc=(开关, 破音临界[源Hz])，内部换算变声后
        if f0_proc and f0_proc[0]:
            f0 = apply_f0_break_protect(f0, f0_proc[1] * pow(2, f0_up_key / 12))

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
            cached = RMVPEExtractor(str(RMVPE_PATH), device, is_half)
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
