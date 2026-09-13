"""推理运行器 — 封装实时推理的缓冲区管理、推理调用、效果器链。

架构重构后：
- 持有 EngineState（与 InferencePipeline 共享，跨块持续状态）
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
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample as TatResample

from rvc.audio.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
from rvc.audio.effects import AudioProcessor
from rvc.audio.output_router import route_secondary_output, write_main_output

logger = logging.getLogger(__name__)


class InferenceRunner:
    """推理运行器 — 管理实时推理的缓冲区与处理流程。

    持有 EngineState（与 InferencePipeline 共享），每块复用 InferenceContext。
    process_block 按 8 阶段闭环执行。
    """

    def __init__(self, pipeline, runtime_params, device: str, function: str = "vc"):
        self.pipeline = pipeline
        self.runtime_params = runtime_params
        self._device = device
        self.function = function

        # 共享 EngineState（pipeline 创建并持有）
        self.state = pipeline.state
        self.state.function = function

        # 单块上下文（复用实例，每块 reset()）
        self.ctx = pipeline.ctx

        # 处理状态（init_processing 后填充）
        self.sr = None
        self.sr_model = None
        self.hz_centis = None
        self.channels = 1
        self.block_samples = 0
        self.block_samples_16k = 0
        self.crossfade_samples = 0
        self.sola_buffer_samples = 0
        self.sola_search_samples = 0
        self.extra_samples = 0
        self.skip_head = 0
        self.return_length = 0

        # 缓冲区（也存在 state 中，这里保留引用方便访问）
        self.input_wav = None
        self.input_wav_res = None
        self.input_wav_work = None
        self.input_wav_res_work = None
        self._in_pin = None

        # 重采样器
        self.resampler = None
        self.resampler_model2dev = None

        # 效果器
        self.processor = AudioProcessor()

        # 性能统计
        self.infer_ms = 0.0

        # 错误状态（音频回调中使用）
        self.error_count = 0
        self.max_error_count = 3
        self.last_error = ""
        self.runtime_error_pending = False

    def init_processing(self, sr: int, block_t: float, cf_t: float, extra_t: float,
                        channels: int, sr_model: int) -> None:
        """初始化所有缓冲区和计算参数。

        Args:
            sr: 工作采样率
            block_t: 块时长（秒）
            cf_t: 交叉淡化时长（秒）
            extra_t: 额外上下文时长（秒）
            channels: 通道数
            sr_model: 模型目标采样率
        """
        self.sr = sr
        self.sr_model = sr_model
        self.channels = channels
        zc = sr // 100
        self.hz_centis = zc

        # 计算参数（对齐到 10ms 边界）
        self.block_samples = int(np.round(block_t * sr / zc)) * zc
        self.crossfade_samples = int(np.round(cf_t * sr / zc)) * zc
        self.sola_buffer_samples = min(self.crossfade_samples, 4 * zc)
        self.sola_search_samples = zc
        self.extra_samples = int(np.round(extra_t * sr / zc)) * zc

        self.block_samples_16k = HUBERT_FRAME_SIZE * self.block_samples // zc
        self.skip_head = self.extra_samples // zc
        self.return_length = (self.block_samples + self.sola_buffer_samples + self.sola_search_samples) // zc

        # 同步到 EngineState
        self.state.target_sr = sr
        self.state.block_samples = self.block_samples
        self.state.crossfade_samples = self.crossfade_samples
        self.state.sola_buffer_samples = self.sola_buffer_samples
        self.state.sola_search_samples = self.sola_search_samples
        self.state.extra_samples = self.extra_samples
        self.state.skip_head = self.skip_head
        self.state.return_length = self.return_length
        self.state.hz_centis = self.hz_centis
        self.state.channels = channels
        self.state.block_samples_16k = self.block_samples_16k

        # 48k 滚动缓冲区
        n = self.extra_samples + self.crossfade_samples + self.sola_search_samples + self.block_samples
        self.input_wav = torch.zeros(n, device=self._device)
        self.input_wav_work = torch.empty_like(self.input_wav)
        self.state.input_wav_48k = self.input_wav

        # 16k 滚动缓冲区（HuBERT / F0 用）
        self.input_wav_res = torch.zeros(HUBERT_FRAME_SIZE * n // zc, device=self._device)
        self.input_wav_res_work = torch.empty_like(self.input_wav_res)
        self.state.input_wav_16k = self.input_wav_res

        # pinned memory 用于快速 CPU→GPU 传输
        self._in_pin = torch.empty(self.block_samples, dtype=torch.float32, pin_memory=True)

        # 重采样器（48k → 16k）
        self.resampler = TatResample(sr, HUBERT_SAMPLE_RATE, dtype=torch.float32).to(self._device)
        self.state.resampler_48k_to_16k = self.resampler

        # 重采样器（模型采样率 → 48k）
        if sr_model != sr:
            self.resampler_model2dev = TatResample(sr_model, sr, dtype=torch.float32).to(self._device)
            self.state.resampler_model_to_48k = self.resampler_model2dev
        else:
            self.resampler_model2dev = None
            self.state.resampler_model_to_48k = None

        # 效果器（RMS + SOLA）
        self.processor.setup(
            sr, self.block_samples, self.crossfade_samples,
            self.sola_search_samples, self._device,
        )
        self.state.audio_processor = self.processor

        # SOLA 输出缓冲区
        self.state.sola_buffer = torch.zeros(self.sola_buffer_samples, device=self._device, dtype=torch.float32)

    def warmup(self, n: int = 2) -> None:
        """预热推理引擎（用静音数据跑 n 次，捕获 CUDA Graph）。

        失败只警告不影响运行。
        """
        if self.pipeline is None or self.input_wav_res is None:
            return
        frames = self.block_samples
        indata = np.zeros((frames, self.channels), dtype=np.float32)
        outdata = np.zeros((frames, self.channels), dtype=np.float32)
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
        if self.pipeline is not None:
            self.pipeline.reset_pitch_cache()
        self.processor.reset()
        if self.input_wav is not None:
            self.input_wav.zero_()
        if self.input_wav_res is not None:
            self.input_wav_res.zero_()

    def reset_error_state(self) -> None:
        self.error_count = 0
        self.last_error = ""
        self.runtime_error_pending = False

    def reset_success_count(self) -> None:
        self.error_count = 0

    def handle_error(self, error: Exception) -> bool:
        self.error_count += 1
        self.last_error = str(error)
        if self.error_count >= self.max_error_count and not self.runtime_error_pending:
            self.runtime_error_pending = True
            return True
        return False

    # ── 主处理入口 ──

    def process_block(self, indata: np.ndarray, outdata: np.ndarray, frames: int) -> None:
        """处理一块音频，执行完整的 8 阶段闭环，直接写入 outdata。

        Args:
            indata: 输入音频（numpy array，shape=(frames, channels)）
            outdata: 输出音频缓冲区（numpy array，shape=(frames, channels)）
            frames: 帧数
        """
        t0 = time.perf_counter()
        params = self.runtime_params
        p_rms_mix = params.rms_mix

        with torch.no_grad():
            # 阶段1：硬件输入（单声道化）
            mono = self._stage_input(indata)

            # 阶段2：输入预处理（缓冲区滚动 + 48k→16k 重采样）
            self._stage_preprocess(mono)

            # 阶段3-6：HuBERT → F0 → 合成预处理 → 合成（委托给 pipeline）
            infer = self._stage_inference()

            # 阶段7：输出处理（RMS 匹配 + SOLA 拼接）
            ref = self.input_wav[self.extra_samples:]
            chunk = self.processor.process_output(infer, ref, p_rms_mix, self.function == "vc")

            # 阶段8：硬件输出（写入 outdata）
            write_main_output(chunk, outdata, self.channels)

        self.infer_ms = (time.perf_counter() - t0) * 1000

    def route_secondary_output(self, outdata: np.ndarray, stream2, out2_q: queue.Queue,
                                enable_out2: bool) -> None:
        """副输出路由（委托给 output_router）。"""
        route_secondary_output(outdata, stream2, out2_q, enable_out2)

    # ── 阶段1：硬件输入 ──

    def _stage_input(self, indata: np.ndarray) -> torch.Tensor:
        """阶段1：从声卡获取原始音频，单声道化 + CPU→GPU 传输。

        使用 pinned memory 加速传输。
        """
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        mono = np.ascontiguousarray(mono)
        n = mono.shape[0]
        self._in_pin[:n].copy_(torch.from_numpy(mono), non_blocking=True)
        return self._in_pin[:n].to(self._device, non_blocking=True)

    # ── 阶段2：输入预处理 ──

    def _stage_preprocess(self, mono: torch.Tensor) -> None:
        """阶段2：缓冲区滚动 + 48k→16k 重采样。

        使用双缓冲交换避免额外拷贝。16k 重采样只处理新块（带额外上下文抵消重采样延迟）。
        """
        # 48k 缓冲区左移（双缓冲交换）
        self.input_wav_work[:-self.block_samples].copy_(self.input_wav[self.block_samples:])
        self.input_wav_work[-self.block_samples:].zero_()
        self.input_wav, self.input_wav_work = self.input_wav_work, self.input_wav
        self.input_wav[-mono.shape[0]:] = mono

        # 16k 缓冲区左移（双缓冲交换）
        self.input_wav_res_work[:-self.block_samples_16k].copy_(self.input_wav_res[self.block_samples_16k:])
        self.input_wav_res_work[-self.block_samples_16k:].zero_()
        self.input_wav_res, self.input_wav_res_work = self.input_wav_res_work, self.input_wav_res

        # 只重采样新块（带额外 2*hz_centis 上下文抵消重采样延迟），输出跳过前 HUBERT_FRAME_SIZE 样本
        resampler_in = self.input_wav[-mono.shape[0] - 2 * self.hz_centis:]
        resampler_out = self.resampler(resampler_in)[HUBERT_FRAME_SIZE:]
        target_len = HUBERT_FRAME_SIZE * (mono.shape[0] // self.hz_centis + 1)
        self.input_wav_res[-target_len:] = resampler_out

        # 同步到 EngineState
        self.state.input_wav_48k = self.input_wav
        self.state.input_wav_16k = self.input_wav_res

    # ── 阶段3-6：推理（委托给 pipeline） ──

    def _stage_inference(self) -> torch.Tensor:
        """阶段3-6：HuBERT → F0 → 合成预处理 → 合成。

        委托给 InferencePipeline.infer，然后做模型→设备重采样和长度对齐。
        """
        if self.function == "vc" and self.pipeline:
            infer = self.pipeline.infer(
                self.input_wav_res, self.runtime_params,
                self.block_samples_16k, self.skip_head, self.return_length,
            )
            # 模型采样率 → 工作采样率 重采样（如果需要）
            if self.resampler_model2dev:
                infer = self.resampler_model2dev(infer)
        else:
            # 直通模式（非 vc）：直接用原始输入
            infer = self.input_wav[self.extra_samples:].clone()

        # 长度对齐
        expected = self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        if infer.shape[0] > expected:
            infer = infer[:expected]
        elif infer.shape[0] < expected:
            infer = F.pad(infer, (0, expected - infer.shape[0]))

        return infer
