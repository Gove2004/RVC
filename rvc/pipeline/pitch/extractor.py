"""F0 提取器抽象层 — 统一 RMVPE 和 FCPE 的接口。

后处理流程（简化后）：
  原始 F0 → 音域映射（半音尺度）→ 因果中值滤波 → 离散化

"""
import contextlib
import gc
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from rvc.core.constants import FCPE_TAIL_WINDOW_SEC, HUBERT_FRAME_SIZE
from rvc.core.errors import F0ExtractionError
from rvc.pipeline.pitch.fcpe_mel import FCPEMelExtractor
from rvc.pipeline.pitch.postprocess import apply_pitch_map, median_filter_f0, normalize_f0_to_coarse
from rvc.runtime.graph import clear_cuda_graph_cache, cuda_graph_enabled, run_cuda_graph
from rvc.runtime.paths import RMVPE_PATH

# 最新原始输入音调（Hz，非零帧平均，音域映射之前的值，用于 GUI 显示）

logger = logging.getLogger(__name__)


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


def postprocess_f0(f0, device, confidence=None, config=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把提取器原始 F0 统一后处理为 (pitch_coarse, pitchf, confidence)。

    流程：音域映射（半音尺度）→ 因果中值滤波 → 离散化。

    Args:
        f0: 原始连续 F0（可能是 np.ndarray 或 tensor，Hz）
        device: 目标设备
        confidence: 逐帧置信度（0~1），None 时用 f0>0 伪置信度
        config: InferenceParams（含 pitch_map_* 音域映射参数）

    Returns:
        (pitch_coarse, pitchf, confidence):
        离散 pitch、连续 pitch（映射+滤波后）、置信度
    """
    if config is None:
        from rvc.core.config import InferenceParams
        config = InferenceParams()
    if not torch.is_tensor(f0):
        f0 = torch.from_numpy(f0)
    f0 = f0.float().to(device).squeeze()

    # confidence 因果中值滤波（kernel=3）：避免孤立低值帧
    if confidence is not None:
        if not torch.is_tensor(confidence):
            confidence = torch.from_numpy(confidence)
        confidence = confidence.float().to(device).squeeze()
        confidence = median_filter_f0(confidence, kernel=3)

    # 音域映射（半音尺度，始终生效）
    f0 = apply_pitch_map(
        f0,
        config.voice.pitch_map_src_min,
        config.voice.pitch_map_src_max,
        config.voice.pitch_map_dst_min,
        config.voice.pitch_map_dst_max,
    )

    # F0 因果中值滤波（kernel=3）：消除孤立误判帧，减少气声和抖动
    f0 = median_filter_f0(f0, kernel=3)

    # confidence 为 None 时用滤波后 F0 生成伪置信度
    if confidence is None:
        confidence = (f0 > 0).float()

    return normalize_f0_to_coarse(f0), f0, confidence


class F0Extractor(ABC):
    """F0 提取器抽象基类 — 统一接口。"""

    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取 F0 (pitch_coarse, pitchf, confidence)。

        Args:
            audio: 输入音频 (1D Tensor)
            sr: 采样率

        Returns:
            (pitch_coarse, pitchf, confidence):
            离散 pitch、连续 pitch、逐帧置信度
        """
        pass

    def extract_raw(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        """只提取原始 F0 和 confidence，不做后处理（音域映射/滤波/离散化）。

        默认实现：调用 extract 后从 pitchf 反推原始值（不准确）。
        子类应重写此方法以提供真正的原始提取。

        Args:
            audio: 输入音频 (1D Tensor)
            sr: 采样率

        Returns:
            (f0_raw, confidence_raw): 原始连续 F0 (Hz)、原始置信度
        """
        _, pitchf, conf = self.extract(audio, sr)
        return pitchf, conf

    @abstractmethod
    def clear_cuda_graph(self) -> None:
        """清除该提取器的 CUDA Graph 缓存（模型切换/重启时调用）。"""
        pass

    def release(self) -> None:
        """完全释放提取器：清除 CUDA Graph + 删除模型 + 回收显存。

        切换 F0 方法时调用，避免旧模型驻留显存。
        模板方法：clear_cuda_graph() → _release_model() → gc.collect() → empty_cache()。
        子类只需重写 _release_model() 删除自身模型属性。
        """
        self.clear_cuda_graph()
        self._release_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_model(self) -> None:
        """子类重写：删除自身模型属性。默认空实现。"""
        pass


class RMVPEExtractor(F0Extractor):
    """RMVPE F0 提取器"""

    def __init__(self, model_path: str, device: torch.device, is_half: bool, config=None) -> None:
        from rvc.models.rmvpe import RMVPE
        mp = Path(model_path)
        if not mp.exists():
            raise F0ExtractionError(
                f"RMVPE 权重文件不存在: {mp}\n"
                f"请将 rmvpe.pt 放到 {RMVPE_PATH.parent}/ 下，"
                f"或参考 assets/README/RVC.md 中的指引。"
            )
        logger.info("加载 RMVPE")
        self.model = RMVPE(mp, is_half=is_half, device=device)
        self.device = device
        self.config = config  # InferenceParams，含 rmvpe_threshold

    def extract_raw(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        """只提取原始 F0 和 confidence，不做后处理。"""
        from rvc.core.config import InferenceParams
        params = self.config or InferenceParams()
        f0, conf = self.model.infer_from_audio_with_confidence(audio, thred=params.f0.rmvpe_threshold)
        return f0, conf

    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0, conf = self.extract_raw(audio, sr)
        return postprocess_f0(f0, self.device, confidence=conf, config=self.config)

    def clear_cuda_graph(self) -> None:
        clear_cuda_graph_cache(self.model.mel_extractor)
        clear_cuda_graph_cache(self.model)

    def _release_model(self) -> None:
        self.model = None


class FCPEExtractor(F0Extractor):
    """FCPE F0 提取器"""

    def __init__(self, device: torch.device, config=None) -> None:
        from torchfcpe import spawn_bundled_infer_model
        logger.info("加载 FCPE")
        self.config = config  # InferenceParams，含 fcpe_confidence_threshold
        fcpe_logger = logging.getLogger("torchfcpe")
        saved_level = fcpe_logger.level
        fcpe_logger.setLevel(logging.ERROR)
        try:
            with _suppress_torchfcpe_output():
                self.model = spawn_bundled_infer_model(device)
        finally:
            fcpe_logger.setLevel(saved_level)

        # 手动解码（含 confidence 计算）依赖 torchfcpe 的模型核心 `infer_model.model`；
        # 缺失时降级到官方 self.model.infer（不需要 local_offsets，也没有 confidence）。
        self._has_model_core = hasattr(self.model, "model")
        # 仓库内复刻的 mel 前端（见 rvc/pipeline/pitch/fcpe_mel.py）：去掉 torchfcpe
        # MelModule 每块两次设备同步，且可被 CUDA Graph 捕获。torchfcpe 布局变化
        # 导致取不到 mel_extractor 时置 None，由 _mel() 退回原实现并告警。
        self._has_mel_frontend = (
            hasattr(self.model, "wav2mel") and hasattr(self.model.wav2mel, "mel_extractor")
        )
        self.mel = (
            FCPEMelExtractor(self.model.wav2mel).to(device).eval()
            if self._has_mel_frontend else None
        )
        self._mel_fallback_warned = False
        # 局部窗口偏移表：图捕获与非图手动解码都要求它常驻设备，故不再与
        # 「CUDA Graph 是否可用」绑定（旧逻辑只在图可用时创建，导致非图路径
        # 一旦需要它就直接 AttributeError）。
        self.local_offsets = (
            torch.arange(9, device=device, dtype=torch.long).view(1, 1, 9)
            if self._has_model_core else None
        )
        self.device = device

    def _mel(self, wav_t: torch.Tensor, sr: int) -> torch.Tensor:
        """wav → mel 前端。

        采样率与模型一致且复刻可用时，走仓库内复刻（**无设备同步**）；
        其余情形退回 torchfcpe 原实现（含其内部重采样），并告警一次——那里每块
        含两次 GPU→CPU 同步，属于降级而非等价路径。

        ⚠️ **刻意不做 CUDA Graph 捕获**（尽管复刻实现本身是可捕获的）：
        本调用发生在 F0 的 side stream 上，捕获后它会在侧流回放，与主流上的
        HuBERT / 合成器图回放处于不同私有显存池。实测（真实实时链路，RTX 5060
        Laptop）这样跑约 30s 后音频回调会卡死在 `apply_pitch_map` 的 GPU→CPU
        同步上（设备再也排不空 → 回调不再被调用 → GUI 冻结、GPU 占满）。
        图化相对 eager 只省约 1.66ms（0.53 vs 2.19ms），而代价是整条实时链路
        不稳定——真正值钱的收益是「去掉 torchfcpe MelModule 每块两次设备同步」
        （3.31 → 2.19ms），这一项无副作用地保留了。
        """
        if self.mel is not None and sr == self.mel.sample_rate:
            return self.mel(wav_t)
        if not self._mel_fallback_warned:
            logger.warning(
                "FCPE mel 前端退回 torchfcpe 原实现（sr=%s，复刻 sample_rate=%s）："
                "该路径每块含两次设备同步", sr, self.mel.sample_rate if self.mel else None,
            )
            self._mel_fallback_warned = True
        return self.model.wav2mel(wav_t, sr)

    def extract_raw(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        """只提取原始 F0 和 confidence，不做后处理。

        返回值的**帧坐标契约**：下标 j 对应输入音频第 j 个 10ms 帧（即样本 j*160），
        与传入的整段 audio 一致。用尾部窗口时靠前导零补回原坐标（见下），
        调用方（_sync_f0_parallel）因此不需要知道窗口的存在。
        """
        from rvc.core.config import InferenceParams
        params = self.config or InferenceParams()

        # F0 参考窗与 HuBERT 的上下文窗无关：F0 网络是局部感受野、后处理是因果中值
        # 滤波，而最终只有 p_len 末尾的 return_length 帧进合成器——前段 F0 永远用不上，
        # 算它纯属浪费。窗口足够长时结果逐帧等效（见 FCPE_TAIL_WINDOW_SEC 的实测表）。
        #
        # 窗口长度对齐到 10ms 帧网格，保证截断后的 mel 帧栅格与原输入完全重合
        # （否则会引入亚帧偏移）。随后把结果左补 skip_frames 个零**补回原始帧坐标**：
        # 这一步不能省——_sync_f0_parallel 按 `[:p_len]` 左对齐取样，若只补到 p_len，
        # F0 会整体错位 (跳过帧数 - (p_len - n)) 帧（实测 3 帧 = 30ms，在快颤音上
        # 表现为 ~0.18 半音的偏差）。
        total = audio.shape[-1]
        n_tail = (int(sr * FCPE_TAIL_WINDOW_SEC) // HUBERT_FRAME_SIZE) * HUBERT_FRAME_SIZE
        skip_frames = 0
        if n_tail > 0 and total > n_tail:
            skip_frames = (total - n_tail) // HUBERT_FRAME_SIZE
            audio = audio[-n_tail:]

        wav_t = audio.to(self.device).unsqueeze(0).float()

        with _suppress_torchfcpe_output():
            if cuda_graph_enabled(self.device) and self._has_model_core:
                mel = self._mel(wav_t, sr)

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
                    confidence_mask.masked_fill_(confidence <= params.f0.fcpe_confidence_threshold, float("-inf"))
                    decoded = decoded * confidence_mask
                    f0 = 10.0 * torch.pow(2.0, decoded / 1200.0)
                    return f0, confidence.squeeze(-1)

                f0, conf = run_cuda_graph(
                    self.model.model,
                    f"fcpe-core-local_argmax-conf-{params.f0.fcpe_confidence_threshold}",
                    graphable_infer, mel,
                )
            else:
                f0 = self.model.infer(
                    wav_t, sr=sr, decoder_mode="local_argmax",
                    threshold=params.f0.fcpe_confidence_threshold,
                )
                conf = None
        f0_out = f0.squeeze() if f0.dim() > 1 else f0
        conf_out = conf.squeeze() if conf is not None and conf.dim() > 1 else conf
        if skip_frames:
            # 前导零补回原始输入帧坐标——调用方按 `[:p_len]` 左对齐取样，
            # 少了这一步 F0 会整体前移 skip_frames 帧（见 docstring 的帧坐标契约）。
            f0_out = torch.nn.functional.pad(f0_out, (skip_frames, 0))
            if conf_out is not None:
                conf_out = torch.nn.functional.pad(conf_out, (skip_frames, 0))
        return f0_out, conf_out

    def extract(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0, conf = self.extract_raw(audio, sr)
        return postprocess_f0(f0, self.device, confidence=conf, config=self.config)

    def clear_cuda_graph(self) -> None:
        if self._has_model_core:
            clear_cuda_graph_cache(self.model.model)
        if self.mel is not None:
            clear_cuda_graph_cache(self.mel)

    def _release_model(self) -> None:
        self.model = None
        self.mel = None
        self.local_offsets = None


def create_f0_extractor(method: str, device: torch.device, is_half: bool, inference_cache, config=None) -> F0Extractor:
    """F0 提取器工厂函数 — 支持缓存。

    注意：不要在此处清除 CUDA Graph！本函数每次推理都会被调用，
    清除 CUDA Graph 会导致图被反复重建，推理卡顿。
    CUDA Graph 缓存在模型加载时（ModelSessions）统一清除。
    """
    if method == "rmvpe":
        # 方法切换时释放 FCPE（仅在真正变化时执行，内部有 _active_f0_method 标记）
        inference_cache.switch_f0_method("rmvpe")
        cache_key = (device, is_half)
        cached = inference_cache.get_rmvpe(cache_key)
        if cached is None:
            cached = RMVPEExtractor(str(RMVPE_PATH), device, is_half, config=config)
            inference_cache.set_rmvpe(cache_key, cached)
        return cached
    elif method == "fcpe":
        # 方法切换时释放 RMVPE（仅在真正变化时执行）
        inference_cache.switch_f0_method("fcpe")
        cache_key = device
        cached = inference_cache.get_fcpe(cache_key)
        if cached is None:
            cached = FCPEExtractor(device, config=config)
            inference_cache.set_fcpe(cache_key, cached)
        return cached
    else:
        raise F0ExtractionError(f"未知的 F0 提取方法: {method}")
