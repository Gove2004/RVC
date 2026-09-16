"""推理运行器 — 封装实时推理的缓冲区管理、推理调用、效果器链。

架构重构后（5阶段）：
- 所有跨块持续状态统一放在 EngineState（与 InferencePipeline 共享）
- InferenceRunner 只持有 pipeline/runtime_params 引用，不重复持有状态字段
- 每块复用 InferenceContext（单块临时状态）
- process_block 按角色名阶段执行（全局唯一一套阶段命名）：
    stage_input：硬件输入 + 缓冲区滚动 + 48k→16k 重采样
    stage_features：HuBERT 特征提取（pipeline.extract_features）
    stage_f0：F0 原始提取（与 stage_features 同步进行）与后处理（postprocess_features）
    stage_synthesis：合成器推理 + 模型→工作采样率重采样 + formant + 长度对齐
    stage_output：RMS 混合 + SOLA + 硬件输出
- formant 因子在 pipeline.extract_features 中计算并存到 ctx.formant_factor，本类直接读取避免重复计算
- 清辅音保护的 pitchf 截取封装在 AudioProcessor 内部，本类只传 pitchf_cache

从 RealtimeEngine 中拆分出的推理调度组件，负责：
- 处理状态初始化（采样率/块大小/缓存/重采样/效果器）
- 输入准备（单声道转换、缓存轮换）
- 推理调用（InferencePipeline 三阶段 + 模型→设备重采样）
- 输出处理（RMS 混合、清辅音保护、SOLA 对齐）
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

from rvc.core.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
from rvc.streaming.output import route_secondary_output, write_main_output
from rvc.streaming.loudness import AudioProcessor
from rvc.pipeline.synthesis import apply_formant_resample

logger = logging.getLogger(__name__)


class InferenceRunner:
    """推理运行器 — 管理实时推理的缓冲区与处理流程。

    所有跨块持续状态放在 EngineState（与 pipeline 共享），本类不重复持有。
    process_block 按角色名阶段闭环执行：stage_input → stage_features/stage_f0
    → stage_synthesis → stage_output。
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

        # 输出效果器（RMS 混合 + SOLA）由 runner 直接持有：
        # 它们是输出侧效果，不属于推理状态（D2：pipeline 产物止于合成音频块）
        self.effects = AudioProcessor()

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
        state.work_sr = sr
        state.model_sr = sr_model
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

        # 效果器（RMS + SOLA）；sola_buffer_samples 已在上面统一计算（A2 单源）
        self.effects.setup(
            sr, state.block_samples, state.sola_buffer_samples,
            state.sola_search_samples, state.device,
        )

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
        self.effects.reset()
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

    # ── 主处理入口（五阶段闭环） ──

    def process_block(self, indata: np.ndarray, outdata: np.ndarray, frames: int) -> None:
        """处理一块音频，执行完整的五阶段闭环，直接写入 outdata。

        stage_input：硬件输入 + 缓冲区滚动 + 48k→16k 重采样
        stage_features / stage_f0：HuBERT 特征 + F0 原始提取
        （F0 后处理 + 特征上采样在 postprocess_features）
        stage_synthesis：合成器推理 + 模型→工作采样率重采样 + formant + 长度对齐
        stage_output：RMS 混合 + SOLA + 硬件输出

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
            # ── stage_input：硬件输入 + 缓冲区滚动 + 48k→16k 重采样 ──
            # 有预取数据时直接用，跳过 CPU→GPU 传输（离线推理预取和计算并行）
            if state.prefetch_valid and state.prefetch_gpu is not None:
                mono = state.prefetch_gpu
                state.prefetch_valid = False
            else:
                mono = self._stage_input(indata)
            self._stage_preprocess(mono)

            # ── stage_features / stage_f0：HuBERT 特征 + F0 原始提取 ──
            if state.function == "vc" and self.pipeline:
                # ctx 生命周期由 runner 管理，每块开始时重置（pipeline 不再负责 reset）
                ctx.reset(self.runtime_params, state.block_samples_16k, state.block_samples)
                feats = self.pipeline.extract_features(
                    state.input_wav_16k, self.runtime_params,
                    state.block_samples_16k, state.skip_head, state.return_length,
                )
            else:
                feats = None

            # ── stage_f0 后处理（F0 后处理 + 特征上采样） ──
            if feats is not None:
                pitch, pitchf, feats = self.pipeline.postprocess_features(feats)
            else:
                pitch, pitchf = None, None

            # ── stage_synthesis：合成器推理 + 模型→工作采样率重采样 ──
            if feats is not None:
                infer = self.pipeline.synthesize_audio(feats, pitch, pitchf)
                # 模型采样率 → 工作采样率 重采样（如果需要）
                if state.resampler_model_to_48k:
                    infer = state.resampler_model_to_48k(infer)
            else:
                # 直通模式（非 vc）：直接用原始输入
                infer = state.input_wav_48k[state.extra_samples:].clone()

            # formant 重采样 + 长度对齐（formant 因子从 ctx.formant_factor 读取，避免重复计算）
            infer = self._stage_formant(infer)
            ctx.formanted_audio = infer

            # ── stage_output：RMS + SOLA + 硬件输出 ──
            ref = state.input_wav_48k[state.extra_samples:]
            chunk = self.effects.process_output(
                infer, ref, p_rms_mix, state.function == "vc",
            )
            ctx.final_output = chunk

            # 硬件输出（写入 outdata）
            write_main_output(chunk, outdata, state.channels)
            ctx.output_np = outdata

        state.infer_ms = (time.perf_counter() - t0) * 1000

    def prefetch_input(self, indata: np.ndarray) -> None:
        """预取下一块输入（只做 stage_input 的输入上传部分）。

        在离线推理中，当前块做 GPU 计算时，提前把下一块输入拷贝到 GPU，
        下一次 process_block 时直接用预取数据，跳过 CPU→GPU 传输等待。

        实时推理中不调用此方法（下一块输入还没到达），prefetch_valid 保持 False。
        """
        state = self.state
        if state.in_pin is None:
            return
        self._upload_input(indata)
        state.prefetch_valid = True

    def route_secondary_output(self, outdata: np.ndarray, stream2, out2_q: queue.Queue,
                                enable_out2: bool) -> None:
        """副输出路由（委托给 output_router）。"""
        route_secondary_output(outdata, stream2, out2_q, enable_out2)

    # ── stage_input：硬件输入 ──

    def _upload_input(self, indata: np.ndarray) -> torch.Tensor:
        """单声道化 + pinned 中转 + CPU→GPU 上传（A7：prefetch 与 stage_input 共用）。

        返回 GPU 上的单声道张量。传输为异步（non_blocking），消费方在同一
        CUDA Stream 上，流内顺序保证安全。
        """
        state = self.state
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        mono = np.ascontiguousarray(mono)
        n = mono.shape[0]
        state.in_pin[:n].copy_(torch.from_numpy(mono), non_blocking=True)
        if state.prefetch_gpu is None or state.prefetch_gpu.shape[0] < n:
            state.prefetch_gpu = torch.empty(n, device=state.device, dtype=torch.float32)
        state.prefetch_gpu[:n].copy_(state.in_pin[:n], non_blocking=True)
        return state.prefetch_gpu[:n]

    def _stage_input(self, indata: np.ndarray) -> torch.Tensor:
        """stage_input：硬件输入 → GPU（复用 _upload_input）。"""
        return self._upload_input(indata)

    # ── stage_input：输入预处理（缓冲区滚动 + 48k→16k 重采样） ──

    def _stage_preprocess(self, mono: torch.Tensor) -> None:
        """缓冲区滚动 + 48k→16k 重采样。

        使用双缓冲交换避免额外拷贝。16k 重采样只处理新块（带额外上下文抵消重采样延迟）。
        """
        state = self.state

        # 16k 缓冲区左移（双缓冲交换）— 与 48k 滚动互不依赖，顺序可交换
        state.input_wav_16k_work[:-state.block_samples_16k].copy_(state.input_wav_16k[state.block_samples_16k:])
        state.input_wav_16k_work[-state.block_samples_16k:].zero_()
        state.input_wav_16k, state.input_wav_16k_work = state.input_wav_16k_work, state.input_wav_16k

        # 48k 缓冲区左移（双缓冲交换）
        state.input_wav_48k_work[:-state.block_samples].copy_(state.input_wav_48k[state.block_samples:])
        state.input_wav_48k_work[-state.block_samples:].zero_()
        state.input_wav_48k, state.input_wav_48k_work = state.input_wav_48k_work, state.input_wav_48k
        state.input_wav_48k[-mono.shape[0]:] = mono

        # 只重采样新块（带额外 2*hz_centis 上下文抵消重采样延迟），输出跳过前 HUBERT_FRAME_SIZE 样本
        resampler_in = state.input_wav_48k[-mono.shape[0] - 2 * state.hz_centis:]
        resampler_out = state.resampler_48k_to_16k(resampler_in)[HUBERT_FRAME_SIZE:]
        # 只写当前块（最后 block_samples_16k 样本），不写修正帧（前 160 样本）。
        # 修正帧会覆盖上一块最后10ms，导致 HuBERT 看到的历史在每次推理中都被"修正"，
        # 连续累积后特征漂移（如"凤凰"连续读越读越飘成"废黄"）。
        target_len = state.block_samples_16k
        state.input_wav_16k[-target_len:] = resampler_out[-target_len:]

    # ── stage_synthesis：formant 重采样 + 长度对齐 ──

    def _stage_formant(self, infer: torch.Tensor) -> torch.Tensor:
        """formant 共振峰偏移（重采样实现）+ 长度对齐。

        formant 因子从 ctx.formant_factor 读取（pipeline.extract_features 中已计算），
        避免重复计算 pow(2, formant/12)。
        formant=0 时直接做长度对齐返回。
        formant 重采样后统一对齐到 expected 长度（SOLA 期望固定长度）。

        Args:
            infer: 合成器输出音频（target_sr 采样率）

        Returns:
            formant 处理 + 长度对齐后的音频
        """
        state = self.state
        ctx = self.ctx
        expected = state.block_samples + state.sola_buffer_samples + state.sola_search_samples

        # formant 因子从 ctx 读取（pipeline.extract_features 中已计算并存入）
        formant = self.runtime_params.voice.formant
        if formant != 0:
            factor = ctx.formant_factor
            target_sr = state.work_sr
            upp_res = int(math.floor(factor * target_sr // 100))
            if upp_res != target_sr // 100:
                infer = apply_formant_resample(
                    infer[: state.return_length * upp_res],
                    factor, target_sr, state.resample_kernel, state.device,
                ).squeeze()

        # 统一长度对齐（formant 重采样可能改变长度，SOLA 期望固定长度）
        if infer.shape[0] > expected:
            infer = infer[:expected]
        elif infer.shape[0] < expected:
            infer = F.pad(infer, (0, expected - infer.shape[0]))
        return infer
