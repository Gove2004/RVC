"""推理管线跨块持续状态 — 启动时创建，关闭时销毁，每块推理时复用。

所有需要在块与块之间保持的数据都放这里：
- 模型引用（load 后注入）
- 48k / 16k 滚动缓冲区
- 合成器缓存（resample_kernel / long_tensor）
- F0 提取器实例

注意：输出效果器（RMS/SOLA，AudioProcessor）不属于推理状态，由
streaming 的 runner 直接持有——pipeline 的产物止于"合成音频块"。

单块临时状态放在 InferenceContext，不在这里。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch


@dataclass
class EngineState:
    """跨块持续状态。

    由 InferencePipeline 创建并持有，InferenceRunner 共享引用，每块推理时使用。
    所有字段均为正式 dataclass 字段，可被 fields()/asdict()/replace() 正确处理。
    """

    # ── 模型引用（load 后注入）──
    hubert_model: Any = None
    synthesizer: Any = None
    f0_extractor: Any = None
    f0_extractor_method: str = ""

    # ── 采样率（语义明确，不再互相覆盖）──
    model_sr: int | None = None   # 模型原生采样率，load() 时设置，只读
    work_sr: int | None = None    # 工作采样率，setup() 时设置
    use_f0: int = 1
    device: str = "cuda"
    is_half: bool = True
    sid: int = 0

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
    hz_centis: int = 0
    channels: int = 1

    # ── 16kHz 滚动缓冲区（HuBERT / F0 用）──
    input_wav_16k: torch.Tensor | None = None
    input_wav_16k_work: torch.Tensor | None = None
    block_samples_16k: int = 0
    resampler_48k_to_16k: Any = None
    resampler_model_to_48k: Any = None

    # ── 输入传输（pinned memory 双缓冲，CPU→GPU 异步 DMA）──
    in_pin: list | None = None  # [pin_a, pin_b] 交替使用
    in_pin_idx: int = 0         # 当前块使用的缓冲索引
    input_gpu: torch.Tensor | None = None  # GPU 侧单声道输入暂存

    # ── 输出传输（pinned memory，GPU→CPU 快速传输，与 in_pin 对称）──
    out_pin: torch.Tensor | None = None

    # ── 合成器相关缓存 ──
    resample_kernel: dict = field(default_factory=dict)
    long_tensor_cache: dict = field(default_factory=dict)

    # ── 模型缓存（ModelCache 实例）──
    inference_cache: Any = None

    # ── 特征检索（index_rate>0 时启用）──
    # index_path 在 pipeline.load() 时由 pth 同目录 <模型名>.index.pt 推导；
    # index_refs 首块 index_rate>0 时懒加载（load_index 缓存命中则零开销）。
    index_path: str = ""
    index_refs: Any = None          # (N, D) 归一化特征张量，CPU 或 GPU
    blend_counter: list = field(default_factory=lambda: [0])  # blend_every_n 计数
    blend_cache: list = field(default_factory=lambda: [None])  # 上次混合结果

    # ── F0 side stream（F0 计算错开在此流上执行，串行于主流的 HuBERT，
    #    调用时机契约见 pipeline._start_f0_parallel）──
    f0_stream: Any = None

    # ── 性能统计 ──
    infer_ms: float = 0.0
    t_input_ms: float = 0.0   # 输入上传 + 48k→16k 重采样
    t_hubert_ms: float = 0.0  # HuBERT 特征提取
    t_f0_ms: float = 0.0      # F0 提取 + 后处理 + 特征上采样
    t_synth_ms: float = 0.0   # 合成器推理 + 模型→工作采样率重采样
    t_output_ms: float = 0.0   # RMS + SOLA + 气流声 + 硬件输出

    # ── 输入音高（GUI 显示用，音域映射之前的原始值）──
    # 热路径只留 GPU 定形统计 (pitch_sum, count)，不做 .item() 强制同步；
    # GUI 低频读取 VoiceEngine.input_pitch 时才做唯一的 GPU→CPU 同步取值。
    last_input_pitch_gpu: tuple | None = None

    # ── 错误状态 ──
    error_count: int = 0
    max_error_count: int = 3
    last_error: str = ""
    runtime_error_pending: bool = False

    def reset_buffers(self) -> None:
        """重置所有缓冲区（warmup 后调用，避免静音数据污染）。

        输出效果器的重置由 runner 负责（效果器不在 EngineState 里）。
        """
        if self.input_wav_48k is not None:
            self.input_wav_48k.zero_()
        if self.input_wav_16k is not None:
            self.input_wav_16k.zero_()

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
