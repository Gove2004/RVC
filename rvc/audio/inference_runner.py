"""推理运行器 — 封装实时推理的缓冲区管理、推理调用、效果器链。

架构重构后：
- 所有跨块持续状态统一放在 EngineState（与 InferencePipeline 共享）
- InferenceRunner 只持有 pipeline/runtime_params 引用，不重复持有状态字段
- 每块复用 InferenceContext（单块临时状态）
- process_block 按阶段执行：阶段1-2（输入+预处理）→ 阶段3-6（pipeline.infer）→ 阶段7-8（输出处理+输出）

从 RealtimeEngine 中拆分出的推理调度组件，负责：
- 处理状态初始化（采样率/块大小/缓存/重采样/效果器）
- 输入准备（单声道转换、缓存轮换）
- 推理调用（InferencePipeline.infer + 模型→设备重采样）
- 输出处理（RMS 混合、SOLA 对齐）
- 预热推理（CUDA Graph 捕获）
- 缓冲区重置

RealtimeEngine 保留对外接口，内部委托给本组件处理推理逻辑。
"""
import logging
import math
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample as TatResample

from rvc.audio.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
from rvc.audio.output_router import route_secondary_output, write_main_output
from rvc.inference.synthesis import apply_formant_resample

logger = logging.getLogger(__name__)


class InferenceRunner:
    """推理运行器 — 管理实时推理的缓冲区与处理流程。

    所有跨块持续状态放在 EngineState（与 pipeline 共享），本类不重复持有。
    process_block 按 8 阶段闭环执行。
    """

    def __init__(self, pipeline, runtime_params, device: str, function: str = "vc"):
        self.pipeline = pipeline
        self.runtime_params = runtime_params

        # 共享 EngineState（pipeline 创建并持有）
        self.state = pipeline.state
        self.state.device = device
        self.state.function = function

        # 单块上下文（复用实例，每块 reset()）
        self.ctx = pipeline.ctx

    def init_processing(self, sr: int, block_t: float, cf_t: float, extra_t: float,
                        channels: int, sr_model: int) -> None:
        """初始化所有缓冲区和计算参数，直接写入 EngineState。

        Args:
            sr: 工作采样率
            block_t: 块时长（秒）
            cf_t: 交叉淡化时长（秒）
            extra_t: 额外上下文时长（秒）
            channels: 通道数
            sr_model: 模型目标采样率
        """
        state = self.state
        state.target_sr = sr
        state.sr_model = sr_model
        state.channels = channels
        zc = sr // 100
        state.hz_centis = zc

        # 计算参数（对齐到 10ms 边界）
        state.block_samples = int(np.round(block_t * sr / zc)) * zc
        state.crossfade_samples = int(np.round(cf_t * sr / zc)) * zc
        state.sola_buffer_samples = min(state.crossfade_samples, 4 * zc)
        state.sola_search_samples = zc
        state.extra_samples = int(np.round(extra_t * sr / zc)) * zc

        state.block_samples_16k = HUBERT_FRAME_SIZE * state.block_samples // zc
        state.skip_head = state.extra_samples // zc
        state.return_length = (state.block_samples + state.sola_buffer_samples + state.sola_search_samples) // zc

        # 48k 滚动缓冲区
        n = state.extra_samples + state.crossfade_samples + state.sola_search_samples + state.block_samples
        state.input_wav_48k = torch.zeros(n, device=state.device)
        state.input_wav_48k_work = torch.empty_like(state.input_wav_48k)

        # 16k 滚动缓冲区（HuBERT / F0 用）
        state.input_wav_16k = torch.zeros(HUBERT_FRAME_SIZE * n // zc, device=state.device)
        state.input_wav_16k_work = torch.empty_like(state.input_wav_16k)

        # pinned memory 用于快速 CPU→GPU 传输
        state.in_pin = torch.empty(state.block_samples, dtype=torch.float32, pin_memory=True)

        # 重采样器（48k → 16k）
        state.resampler_48k_to_16k = TatResample(sr, HUBERT_SAMPLE_RATE, dtype=torch.float32).to(state.device)

        # 重采样器（模型采样率 → 48k）
        if sr_model != sr:
            state.resampler_model_to_48k = TatResample(sr_model, sr, dtype=torch.float32).to(state.device)
        else:
            state.resampler_model_to_48k = None

        # 效果器（RMS + SOLA）
        state.audio_processor.setup(
            sr, state.block_samples, state.crossfade_samples,
            state.sola_search_samples, state.device,
        )

        # SOLA 输出缓冲区
        state.sola_buffer = torch.zeros(state.sola_buffer_samples, device=state.device, dtype=torch.float32)

    def warmup(self, n: int = 2) -> None:
        """预热推理引擎（用静音数据跑 n 次，捕获 CUDA Graph）。

        失败只警告不影响运行。
        """
        state = self.state
        if self.pipeline is None or state.input_wav_16k is None:
            return
        frames = state.block_samples
        indata = np.zeros((frames, state.channels), dtype=np.float32)
        outdata = np.zeros((frames, state.channels), dtype=np.float32)
        try:
            with torch.no_grad():
                for _ in range(n):
                    self.process_block(indata, outdata, frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("预热失败，不影响运行: %s", exc)

    def reset_buffers(self) -> None:
        """重置所有缓冲区（warmup 后调用，避免静音数据污染）。"""
        state = self.state
        if self.pipeline is not None:
            self.pipeline.reset_pitch_cache()
        state.audio_processor.reset()
        if state.input_wav_48k is not None:
            state.input_wav_48k.zero_()
        if state.input_wav_16k is not None:
            state.input_wav_16k.zero_()

    def reset_error_state(self) -> None:
        self.state.reset_error_state()

    def reset_success_count(self) -> None:
        self.state.error_count = 0

    def handle_error(self, error: Exception) -> bool:
        return self.state.handle_error(error)

    # ── 主处理入口 ──

    def process_block(self, indata: np.ndarray, outdata: np.ndarray, frames: int) -> None:
        """处理一块音频，执行完整的 8 阶段闭环，直接写入 outdata。

        Args:
            indata: 输入音频（numpy array，shape=(frames, channels)）
            outdata: 输出音频缓冲区（numpy array，shape=(frames, channels)）
            frames: 帧数
        """
        state = self.state
        ctx = self.ctx
        t0 = time.perf_counter()
        p_rms_mix = self.runtime_params.rms_mix

        with torch.no_grad():
            # 阶段1：硬件输入（单声道化）
            mono = self._stage_input(indata)

            # 阶段2：输入预处理（缓冲区滚动 + 48k→16k 重采样）
            self._stage_preprocess(mono)

            # 阶段3-6：HuBERT → F0 → 合成预处理 → 合成（委托给 pipeline）
            infer = self._stage_inference()

            # 阶段7a：formant 重采样（从 pipeline 层移到引擎层）
            infer = self._stage_formant(infer)
            ctx.formanted_audio = infer

            # 阶段7b+7c：RMS 匹配 + SOLA 拼接
            ref = state.input_wav_48k[state.extra_samples:]
            chunk = state.audio_processor.process_output(infer, ref, p_rms_mix, state.function == "vc")
            ctx.final_output = chunk

            # 阶段8：硬件输出（写入 outdata）
            write_main_output(chunk, outdata, state.channels)
            ctx.output_np = outdata

        state.infer_ms = (time.perf_counter() - t0) * 1000

    def route_secondary_output(self, outdata: np.ndarray, stream2, out2_q: queue.Queue,
                                enable_out2: bool) -> None:
        """副输出路由（委托给 output_router）。"""
        route_secondary_output(outdata, stream2, out2_q, enable_out2)

    # ── 阶段1：硬件输入 ──

    def _stage_input(self, indata: np.ndarray) -> torch.Tensor:
        """阶段1：从声卡获取原始音频，单声道化 + CPU→GPU 传输。

        使用 pinned memory 加速传输。
        """
        state = self.state
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        mono = np.ascontiguousarray(mono)
        n = mono.shape[0]
        state.in_pin[:n].copy_(torch.from_numpy(mono), non_blocking=True)
        return state.in_pin[:n].to(state.device, non_blocking=True)

    # ── 阶段2：输入预处理 ──

    def _stage_preprocess(self, mono: torch.Tensor) -> None:
        """阶段2：缓冲区滚动 + 48k→16k 重采样。

        使用双缓冲交换避免额外拷贝。16k 重采样只处理新块（带额外上下文抵消重采样延迟）。
        """
        state = self.state

        # 48k 缓冲区左移（双缓冲交换）
        state.input_wav_48k_work[:-state.block_samples].copy_(state.input_wav_48k[state.block_samples:])
        state.input_wav_48k_work[-state.block_samples:].zero_()
        state.input_wav_48k, state.input_wav_48k_work = state.input_wav_48k_work, state.input_wav_48k
        state.input_wav_48k[-mono.shape[0]:] = mono

        # 16k 缓冲区左移（双缓冲交换）
        state.input_wav_16k_work[:-state.block_samples_16k].copy_(state.input_wav_16k[state.block_samples_16k:])
        state.input_wav_16k_work[-state.block_samples_16k:].zero_()
        state.input_wav_16k, state.input_wav_16k_work = state.input_wav_16k_work, state.input_wav_16k

        # 只重采样新块（带额外 2*hz_centis 上下文抵消重采样延迟），输出跳过前 HUBERT_FRAME_SIZE 样本
        resampler_in = state.input_wav_48k[-mono.shape[0] - 2 * state.hz_centis:]
        resampler_out = state.resampler_48k_to_16k(resampler_in)[HUBERT_FRAME_SIZE:]
        target_len = HUBERT_FRAME_SIZE * (mono.shape[0] // state.hz_centis + 1)
        state.input_wav_16k[-target_len:] = resampler_out

    # ── 阶段3-6：推理（委托给 pipeline） ──

    def _stage_inference(self) -> torch.Tensor:
        """阶段3-6：HuBERT → F0 → 合成预处理 → 合成。

        委托给 InferencePipeline.infer，然后做模型→设备重采样和长度对齐。
        """
        state = self.state
        if state.function == "vc" and self.pipeline:
            infer = self.pipeline.infer(
                state.input_wav_16k, self.runtime_params,
                state.block_samples_16k, state.skip_head, state.return_length,
            )
            # 模型采样率 → 工作采样率 重采样（如果需要）
            if state.resampler_model_to_48k:
                infer = state.resampler_model_to_48k(infer)
        else:
            # 直通模式（非 vc）：直接用原始输入
            infer = state.input_wav_48k[state.extra_samples:].clone()

        # 长度对齐
        expected = state.block_samples + state.sola_buffer_samples + state.sola_search_samples
        if infer.shape[0] > expected:
            infer = infer[:expected]
        elif infer.shape[0] < expected:
            infer = F.pad(infer, (0, expected - infer.shape[0]))

        return infer

    # ── 阶段7a：formant 重采样 ──

    def _stage_formant(self, infer: torch.Tensor) -> torch.Tensor:
        """阶段7a：formant 共振峰偏移（重采样实现）。

        从 pipeline 层移到引擎层，和 RMS/SOLA 统一在输出处理阶段。
        formant=0 时直接返回，不做重采样。

        Args:
            infer: 合成器输出音频（target_sr 采样率）

        Returns:
            formant 处理后的音频
        """
        state = self.state
        formant = self.runtime_params.formant
        if formant == 0:
            return infer

        factor = pow(2, formant / 12)
        target_sr = state.target_sr
        upp_res = int(math.floor(factor * target_sr // 100))
        if upp_res == target_sr // 100:
            return infer

        return apply_formant_resample(
            infer[: state.return_length * upp_res],
            factor, target_sr, state.resample_kernel, state.device,
        ).squeeze()
