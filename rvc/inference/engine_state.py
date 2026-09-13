"""引擎跨块持续状态 — 启动时创建，关闭时销毁，每块推理时复用。

所有需要在块与块之间保持的数据都放这里：
- 模型引用（load 后填充）
- 48k / 16k 滚动缓冲区
- F0 滚动缓存
- 合成器缓存（resample_kernel / long_tensor）
- 输出拼接缓存（sola_buffer / AudioProcessor）
- F0 提取器实例
"""
from dataclasses import dataclass, field

import torch

from rvc.audio.effects import AudioProcessor


@dataclass
class EngineState:
    """跨块持续状态。

    由 InferencePipeline 创建并持有，InferenceRunner 共享引用，每块推理时使用。
    单块临时状态放在 InferenceContext，不在这里。
    """

    # ── 模型引用（load 后填充）──
    hubert_model = None
    synthesizer = None
    target_sr: int | None = None  # 工作采样率（init_processing 后填充）
    sr_model: int | None = None  # 模型目标采样率（init_processing 后填充）
    use_f0: int = 1
    device: str = "cuda"
    is_half: bool = True
    sid: int = 0  # 说话人 ID（从模型卡牌获取）
    function: str = "vc"  # 功能类型（"vc" / 其他）

    # ── 48kHz 滚动缓冲区（引擎层）──
    input_wav_48k: torch.Tensor | None = None
    input_wav_48k_work: torch.Tensor | None = None
    block_samples: int = 0
    crossfade_samples: int = 0
    sola_buffer_samples: int = 0
    sola_search_samples: int = 0
    extra_samples: int = 0
    skip_head: int = 0
    return_length: int = 0
    return_length2: int = 0  # formant 调整后的返回长度
    hz_centis: int = 0
    channels: int = 1

    # ── 16kHz 滚动缓冲区（HuBERT / F0 用）──
    input_wav_16k: torch.Tensor | None = None
    input_wav_16k_work: torch.Tensor | None = None
    block_samples_16k: int = 0
    resampler_48k_to_16k = None
    resampler_model_to_48k = None

    # ── 输入传输（pinned memory，CPU→GPU 快速传输）──
    in_pin: torch.Tensor | None = None

    # ── F0 滚动缓存（pitch_tracker）──
    pitch_cache: torch.Tensor | None = None
    pitchf_cache: torch.Tensor | None = None
    confidence_cache: torch.Tensor | None = None

    # ── 合成器相关缓存 ──
    resample_kernel: dict = field(default_factory=dict)
    long_tensor_cache: dict = field(default_factory=dict)

    # ── 输出拼接缓存（effects）──
    sola_buffer: torch.Tensor | None = None
    audio_processor: AudioProcessor = field(default_factory=AudioProcessor)

    # ── F0 提取器实例（带模型权重，跨块复用）──
    f0_extractor = None

    # ── inference_cache（模型/F0 提取器缓存）──
    inference_cache = None

    # ── 性能统计 ──
    infer_ms: float = 0.0

    # ── 错误状态 ──
    error_count: int = 0
    max_error_count: int = 3
    last_error: str = ""
    runtime_error_pending: bool = False

    def reset_pitch_cache(self) -> None:
        """重置音高缓存（切换模型/文件时调用）。"""
        if self.pitch_cache is not None:
            self.pitch_cache.zero_()
        if self.pitchf_cache is not None:
            self.pitchf_cache.zero_()
        if self.confidence_cache is not None:
            self.confidence_cache.zero_()

    def reset_buffers(self) -> None:
        """重置所有缓冲区（warmup 后调用，避免静音数据污染）。"""
        self.reset_pitch_cache()
        self.audio_processor.reset()
        if self.input_wav_48k is not None:
            self.input_wav_48k.zero_()
        if self.input_wav_16k is not None:
            self.input_wav_16k.zero_()
        if self.sola_buffer is not None:
            self.sola_buffer.zero_()

    def reset_error_state(self) -> None:
        self.error_count = 0
        self.last_error = ""
        self.runtime_error_pending = False

    def handle_error(self, error: Exception) -> bool:
        self.error_count += 1
        self.last_error = str(error)
        if self.error_count >= self.max_error_count and not self.runtime_error_pending:
            self.runtime_error_pending = True
            return True
        return False
