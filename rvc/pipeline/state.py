"""推理管线跨块持续状态 — 启动时创建，关闭时销毁，每块推理时复用。

所有需要在块与块之间保持的数据都放这里：
- 模型引用（load 后注入）
- 工作采样率 / 16k 滚动缓冲区
- 合成器缓存（resample_kernel / long_tensor）
- F0 提取器实例

采样率术语：work_sr = 实时流的工作采样率（可能是设备默认采样率，如 44100/96000，
不必然是 48000），model_sr = 模型原生采样率。所有缓冲区/重采样器都按 work_sr
命名，不要用 "48k" 字样——那在 device 模式下是字面谎言。

注意：输出效果器（RMS/SOLA，AudioProcessor）不属于推理状态，由
streaming 的 runner 直接持有——pipeline 的产物止于"合成音频块"。

单块临时状态放在 InferenceContext，不在这里。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    # ── 采样率（work_sr / model_sr 语义各自独立，互不覆盖）──
    model_sr: int | None = None   # 模型原生采样率，load()/init_processing() 时设置
    work_sr: int | None = None    # 工作采样率，setup() 时设置
    use_f0: bool = True
    device: str = "cuda"
    is_half: bool = True
    sid: int = 0

    # ── work_sr 滚动缓冲区（引擎层）──
    input_wav_work: torch.Tensor | None = None
    input_wav_work_swap: torch.Tensor | None = None
    block_samples: int = 0
    crossfade_samples: int = 0
    sola_buffer_samples: int = 0
    sola_search_samples: int = 0
    extra_samples: int = 0
    skip_head: int = 0
    return_length: int = 0
    samples_per_centis: int = 0  # 每厘秒（10ms）样本数 = sr//100
    channels: int = 1

    # ── 16kHz 滚动缓冲区（HuBERT / F0 用）──
    input_wav_16k: torch.Tensor | None = None
    input_wav_16k_swap: torch.Tensor | None = None
    block_samples_16k: int = 0
    resampler_work_to_16k: Any = None
    resampler_model_to_work: Any = None

    # ── 输入传输（pinned memory 双缓冲，CPU→GPU 异步 DMA）──
    in_pin: list | None = None  # [pin_a, pin_b] 交替使用
    in_pin_np: list | None = None  # 上面两块 pinned 的 numpy 视图（单声道化直接写入）
    in_pin_idx: int = 0         # 当前块使用的缓冲索引
    input_gpu: torch.Tensor | None = None  # GPU 侧单声道输入暂存

    # ── 输出传输（pinned memory，GPU→CPU 快速传输，单缓冲）──
    # 单缓冲即可：D2H 是同步的，写完立刻被复制进设备缓冲，不存在两块在飞。
    out_pin: torch.Tensor | None = None

    # ── 合成器相关缓存 ──
    resample_kernel: dict = field(default_factory=dict)
    long_tensor_cache: dict = field(default_factory=dict)

    # 本块的输入音高（Hz，音域映射前；全清音/无 F0 模型时为 0）。
    # 由 runner 在「同步 D2H 写完主输出之后」取回成 Python float——
    # 那一刻本块的 GPU 队列已经排空，.item() 几乎零等待。所以 GUI 想多快刷
    # 就多快刷，读取这个字段不触发任何 GPU 同步。
    input_pitch_hz: float = 0.0

    # ── 会话级参数（init_processing 时快照，运行中不随 GUI 改动）──
    # formant 因子 2^(formant/12)。它不是「每块读一次配置」的普通参数：
    # 它决定合成器推理图的输出长度（return_length2 进 graph_key），运行中
    # 改变会迫使音频回调线程现场重捕获 CUDA Graph——实测 585ms，相当于
    # 108 个块的耗时，必然断音。故在 init_processing 处快照，改动下次启动生效。
    formant_factor: float = 1.0

    # ── 模型缓存（ModelCache 实例）──
    inference_cache: Any = None

    # ── F0 side stream（F0 计算错开在此流上执行，串行于主流的 HuBERT，
    #    调用时机契约见 pipeline._start_f0_parallel）──
    f0_stream: Any = None

    # ── 性能统计 ──
    # 只留端到端单块耗时：GUI 遥测读它，是唯一有消费者的字段。
    # 需要分阶段排查时用 nsys 采样，别在音频回调里常年挂计时器
    # （每个阶段一次 perf_counter 是热路径上的纯开销）。
    infer_ms: float = 0.0

    # ── 输入音高（GUI 显示用，音域映射之前的原始值）──
    # 热路径只留 GPU 定形统计 (pitch_sum, count)，不做 .item() 强制同步；
    # 取回成 Python float 的动作也在音频线程内完成，写入 input_pitch_hz。
    last_input_pitch_gpu: tuple | None = None

    # ── 错误状态 ──
    error_count: int = 0
    max_error_count: int = 3
    last_error: str = ""
    runtime_error_pending: bool = False

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
