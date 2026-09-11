"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口"""
import contextlib
from rvc.core.errors import F0ExtractionError
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from rvc.runtime.paths import RMVPE_PATH
from rvc.inference.cuda_graph import run_cuda_graph
from rvc.runtime.cuda_graph import cuda_graph_enabled
from rvc.models.rmvpe.constants import F0_MIN, F0_MAX
from rvc.audio.f0_utils import median_filter_f0, normalize_f0_to_coarse, RMVPE_THRESHOLD, apply_pitch_map
from rvc.core.experimental import experimental_config

# 最新原始输入音调（Hz，非零帧平均，音域映射之前的值，用于 GUI 显示）
last_input_pitch = 0.0

logger = logging.getLogger(__name__)

# UV 判定的 confidence 阈值：FCPE 默认 0.006 在低电平底噪（麦克风底噪/呼吸/气声）100%
# 误判浊音给合成器喂假音高，与 RMVPE 的 thred=0.03 拉到同档（RMVPE 在该档底噪全判 uv）。
FCPE_CONFIDENCE_THRESHOLD = 0.025  # 默认值，运行时从 experimental_config 读取


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


# torchfcpe.MelModule 在 |x|>1 时直接 print（不是 logger 调用，无法用 logging 拦截），
# 项目用 _FilteredStream 重定向 sys.stdout/sys.stderr：
#   - prefixes: 行首匹配 → 整行吞掉；
#   - contains: 行内任意位置匹配 → 整行吞掉。
# MelModule 的 print 都是 `[error with torchfcpe.mel_extractor.MelModule]min/max value is ...`，
# 用一行 prefix 即可。MelExtractor（旧版，被某些 torchfcpe 版本调用）的 print 是 `min/max value is ...`
# 不含 `>`，extra-prefix `"min value is "`/`"max value is "` 兼容。
_TORCHFCPE_OUT_PREFIXES = (
    "[INF0]", "[INFO]", "[WARN]",
    "[error with torchfcpe.mel_extractor.MelModule]",
    "min value is ", "max value is ",
)
_TORCHFCPE_OUT_CONTAINS = (
    "torchfcpe.mel_tools.nv_mel_extractor",
    "Librosa not found",
)


@contextlib.contextmanager
def _suppress_third_party_output(*prefixes, contains=()):
    out = _FilteredStream(sys.stdout, prefixes, contains) if sys.stdout else sys.stdout
    err = _FilteredStream(sys.stderr, prefixes, contains) if sys.stderr else sys.stderr
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


def _suppress_torchfcpe_output():
    """包裹整个 FCPE 加载+提取阶段用的上下文，吞掉 MelModule 的 print 噪声。"""
    return _suppress_third_party_output(
        *_TORCHFCPE_OUT_PREFIXES, contains=_TORCHFCPE_OUT_CONTAINS,
    )



def postprocess_f0(f0, device, confidence=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把提取器原始 F0 统一后处理为 (pitch_coarse, pitchf, confidence)。

    RMVPE / FCPE 共用，避免两份重复实现：
    音域映射（半音尺度，始终生效）→ 转 GPU tensor → 离散化。

    Args:
        f0: 原始连续 F0（可能是 np.ndarray 或 tensor，Hz）
        device: 目标设备
        confidence: 逐帧置信度（0~1），None 时用 f0>0 伪置信度

    Returns:
        (pitch_coarse, pitchf, confidence): 离散 pitch、连续 pitch、置信度
    """
    if not torch.is_tensor(f0):
        f0 = torch.from_numpy(f0)
    f0 = f0.float().to(device).squeeze()
    # 保存原始输入音调（音域映射之前，非零帧平均，用于 GUI 显示）
    global last_input_pitch
    nonzero = f0[f0 > 0]
    if nonzero.numel() > 0:
        last_input_pitch = float(nonzero.mean().item())
    # 原始F0（音域映射之前），用于辅音保护的清浊判断
    # （音域映射会改变F0绝对值，导致辅音保护的sigmoid阈值失效）
    f0_raw = f0.clone()
    # 音域映射（半音尺度，始终生效，替代固定 pitch 偏移）
    f0 = apply_pitch_map(
        f0,
        experimental_config.pitch_map_src_min,
        experimental_config.pitch_map_src_max,
        experimental_config.pitch_map_dst_min,
        experimental_config.pitch_map_dst_max,
    )
    # F0中值滤波（kernel=3）：去除孤立误判帧，减少气声和抖动
    f0 = median_filter_f0(f0, kernel=3)
    if confidence is None:
        confidence = (f0 > 0).float()
    else:
        if not torch.is_tensor(confidence):
            confidence = torch.from_numpy(confidence)
        confidence = confidence.float().to(device).squeeze()
    return normalize_f0_to_coarse(f0), f0, confidence, f0_raw


class F0Extractor(ABC):
    """F0 提取器抽象基类 — 统一接口。"""

    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取 F0 (pitch) + confidence。

        Args:
            audio: 输入音频 (1D Tensor)
            sr: 采样率

        Returns:
            (pitch_coarse, pitchf, confidence): 离散 pitch、连续 pitch、逐帧置信度
        """
        pass

    @abstractmethod
    def clear_cuda_graph(self) -> None:
        """清除该提取器的 CUDA Graph 缓存（模型切换/重启时调用）。"""
        pass


class RMVPEExtractor(F0Extractor):
    """RMVPE F0 提取器"""

    def __init__(self, model_path: str, device: torch.device, is_half: bool) -> None:
        from rvc.models.rmvpe import RMVPE
        mp = Path(model_path)
        if not mp.exists():
            # 缺权重时立即抛出明确错误，而不是让 torch.load 在奇怪的 traceback 里炸。
            raise F0ExtractionError(
                f"RMVPE 权重文件不存在: {mp}\n"
                f"请将 rmvpe.pt 放到 {RMVPE_PATH.parent}/ 下，"
                f"或参考 assets/README/RVC.md 中的指引。"
            )
        logger.info("加载 RMVPE")
        self.model = RMVPE(mp, is_half=is_half, device=device)
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0, conf = self.model.infer_from_audio_with_confidence(audio, thred=experimental_config.rmvpe_threshold)
        return postprocess_f0(f0, self.device, confidence=conf)

    def clear_cuda_graph(self) -> None:
        from rvc.inference.cuda_graph import clear_cuda_graph_cache
        clear_cuda_graph_cache(self.model.mel_extractor)
        clear_cuda_graph_cache(self.model)


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
            with _suppress_torchfcpe_output():
                self.model = spawn_bundled_infer_model(device)
        finally:
            fcpe_logger.setLevel(saved_level)

        # CUDA Graph 需要 local_offsets 在 GPU 上（不能在 capture 期间 host→device copy）
        if cuda_graph_enabled(device) and hasattr(self.model, "model"):
            self.local_offsets = torch.arange(9, device=device, dtype=torch.long).view(1, 1, 9)
        else:
            self.local_offsets = None
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wav_t = audio.to(self.device).unsqueeze(0).float()

        # 整个推理包一层 stdout 抑制：wav2mel 内部 MelModule 会在 |x|>1 时 print，
        # 那是高频路径，每帧触发一次。spawn 时只能挡加载期，推理期必须重新拦截。
        with _suppress_torchfcpe_output():
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
                    # experimental_config.fcpe_confidence_threshold（默认 0.025）= RMVPE thred=0.03 同档：
                    # 0.006（torchfcpe 默认）在低电平底噪（麦克风底噪/呼吸/气声）100% 误判浊音，
                    # 给合成器喂假音高。提到 0.025 后底噪全判 uv，与 RMVPE 行为一致。
                    confidence_mask.masked_fill_(confidence <= experimental_config.fcpe_confidence_threshold, float("-inf"))
                    decoded = decoded * confidence_mask
                    f0 = 10.0 * torch.pow(2.0, decoded / 1200.0)
                    return f0, confidence.squeeze(-1)

                f0, conf = run_cuda_graph(
                    self.model.model,
                    f"fcpe-core-local_argmax-conf-{experimental_config.fcpe_confidence_threshold}",
                    graphable_infer, mel,
                )
            else:
                f0 = self.model.infer(
                    wav_t, sr=sr, decoder_mode="local_argmax",
                    threshold=experimental_config.fcpe_confidence_threshold,
                )
                conf = None  # 非 CUDA Graph 路径用 f0>0 伪置信度

        return postprocess_f0(f0, self.device, confidence=conf)

    def clear_cuda_graph(self) -> None:
        from rvc.inference.cuda_graph import clear_cuda_graph_cache
        if hasattr(self.model, "model"):
            clear_cuda_graph_cache(self.model.model)


def create_f0_extractor(method: str, device: torch.device, is_half: bool, inference_cache) -> F0Extractor:
    """F0 提取器工厂函数 — 支持缓存。

    注意：不要在此处清除 CUDA Graph！本函数每次推理都会被调用，
    清除 CUDA Graph 会导致图被反复重建，推理卡顿（声音断断续续）。
    CUDA Graph 缓存在模型加载时（ModelSessionManager）统一清除。
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
        raise F0ExtractionError(f"未知的 F0 提取方法: {method}")


