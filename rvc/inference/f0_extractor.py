"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口"""
import contextlib
import logging
import math
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from rvc.runtime.paths import RMVPE_PATH
from rvc.tools.cuda_graph import cuda_graph_enabled, run_cuda_graph

logger = logging.getLogger(__name__)

# UV 判定的 confidence 阈值：FCPE 默认 0.006 在低电平底噪（麦克风底噪/呼吸/气声）100%
# 误判浊音给合成器喂假音高，与 RMVPE 的 thred=0.03 拉到同档（RMVPE 在该档底噪全判 uv）。
FCPE_CONFIDENCE_THRESHOLD = 0.025

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
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _FilteredStream(stdout, prefixes, contains) if stdout else stdout
    sys.stderr = _FilteredStream(stderr, prefixes, contains) if stderr else stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = stdout, stderr


def _suppress_torchfcpe_output():
    """包裹整个 FCPE 加载+提取阶段用的上下文，吞掉 MelModule 的 print 噪声。"""
    return _suppress_third_party_output(
        *_TORCHFCPE_OUT_PREFIXES, contains=_TORCHFCPE_OUT_CONTAINS,
    )


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
BREAK_PROTECT_DEFAULT_RATIO = 0.4     # 高音区压缩比（越小压越狠；1.0 = 不压缩）
BREAK_PROTECT_DEFAULT_KNEE = 0.12     # 平滑膝宽（相对临界比例：膝范围 = 临界×(1±knee)）


def postprocess_f0(f0, f0_up_key: float, device, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """把提取器原始 F0 统一后处理为 (pitch_coarse, pitchf)。

    RMVPE / FCPE 共用，避免两份重复实现：
    音高偏移 ×2^(key/12) → 转 GPU tensor → 破音保护 → 离散化。

    Args:
        f0: 原始连续 F0（可能是 np.ndarray 或 tensor，Hz）
        f0_up_key: 音高偏移（半音）
        device: 目标设备
        f0_proc: (破音保护开关, 破音临界[源Hz]) 或 None

    Returns:
        (pitch_coarse, pitchf): 离散 pitch 和连续 pitch
    """
    f0 = f0 * pow(2, f0_up_key / 12)
    if not torch.is_tensor(f0):
        f0 = torch.from_numpy(f0)
    f0 = f0.float().to(device).squeeze()
    # 破音保护（变声后域）：f0_proc=(开关, 破音临界[源Hz])，内部换算变声后。
    # 压缩比/膝宽为内部固定默认值（用户无需调节；感觉高音压得不够就把临界 Hz 调低）。
    if f0_proc and f0_proc[0]:
        critical = f0_proc[1] * pow(2, f0_up_key / 12)
        f0 = apply_f0_break_protect(f0, critical)
    return _normalize_f0_to_coarse(f0), f0


def apply_f0_break_protect(f0: torch.Tensor, critical_hz: float,
                           ratio: float = BREAK_PROTECT_DEFAULT_RATIO,
                           knee: float = BREAK_PROTECT_DEFAULT_KNEE) -> torch.Tensor:
    """破音保护（方案 A：压缩比 + 平滑膝）。

    作用在变声后的 F0（Hz）上。三段映射：
      x ≤ C-k        → 原样 y = x（说话/低音区动态 100% 保留）
      C-k ≤ x ≤ C+k  → smoothstep 过渡，斜率从 1 平滑降到 ratio（膝部软拐）
      x > C+k        → y = C + ratio·(x - C)，高音区按比例收窄但仍跟随旋律
    其中 C=临界、k=C·knee（膝半宽，相对临界）、ratio=压缩比。

    与旧版「单指数渐近到 C×1.25」相比：高音区不再被压到一个固定顶，
    而是按 ratio 收窄、仍保留相对旋律，听感更自然。
    ratio/knee 为内部固定默认值（用户只调 critical_hz，保证自动生效）。

    满足：连续 + 一阶连续 + 单调 + 压缩可控。

    Args:
        f0: 变声后的连续 F0 值 (Hz, GPU tensor)
        critical_hz: 破音临界（变声后 Hz）
        ratio: 高音区压缩比（内部默认 0.4）
        knee: 平滑膝宽（内部默认 0.12，相对临界比例）
    """
    if critical_hz <= 0 or f0 is None:
        return f0
    r = max(1e-3, float(ratio))
    C = float(critical_hz)
    if r >= 1.0:
        return f0  # ratio=1：不压缩，原样
    k = max(0.0, float(knee)) * C
    if k <= 1e-6:
        # 膝宽 0：分段线性硬折，斜率 1 → r
        return torch.where(f0 <= C, f0, C + r * (f0 - C))
    lower = C - k
    upper = C + k
    T = (f0 - lower) / (2 * k)          # 膝内 0..1
    intS = T ** 3 - T ** 4 / 2          # ∫ smoothstep
    y_knee = lower + 2 * k * (T + (r - 1) * intS)
    y_hi = C + r * (f0 - C)             # 高音区线性收窄（右段）
    return torch.where(f0 <= lower, f0, torch.where(f0 < upper, y_knee, y_hi))


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
        mp = Path(model_path)
        if not mp.exists():
            # 缺权重时立即抛出明确错误，而不是让 torch.load 在奇怪的 traceback 里炸。
            raise FileNotFoundError(
                f"RMVPE 权重文件不存在: {mp}\n"
                f"请将 rmvpe.pt 放到 {RMVPE_PATH.parent}/ 下，"
                f"或参考 assets/README/RVC.md 中的指引。"
            )
        logger.info("加载 RMVPE")
        self.model = RMVPE(mp, is_half=is_half, device=device)
        self.device = device

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.model.infer_from_audio(audio, thred=0.03)
        return postprocess_f0(f0, f0_up_key, self.device, f0_proc)


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

    def extract(self, audio: torch.Tensor, sr: int, f0_up_key: int, f0_proc: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
                    # FCPE_CONFIDENCE_THRESHOLD（默认 0.025）= RMVPE thred=0.03 同档：
                    # 0.006（torchfcpe 默认）在低电平底噪（麦克风底噪/呼吸/气声）100% 误判浊音，
                    # 给合成器喂假音高。提到 0.025 后底噪全判 uv，与 RMVPE 行为一致。
                    confidence_mask.masked_fill_(confidence <= FCPE_CONFIDENCE_THRESHOLD, float("-inf"))
                    decoded = decoded * confidence_mask
                    return 10.0 * torch.pow(2.0, decoded / 1200.0)

                f0 = run_cuda_graph(
                    self.model.model,
                    f"fcpe-core-local_argmax-{FCPE_CONFIDENCE_THRESHOLD}",
                    graphable_infer, mel,
                )
            else:
                f0 = self.model.infer(
                    wav_t, sr=sr, decoder_mode="local_argmax",
                    threshold=FCPE_CONFIDENCE_THRESHOLD,
                )

        return postprocess_f0(f0, f0_up_key, self.device, f0_proc)


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
