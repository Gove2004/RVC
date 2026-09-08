"""推理运行器 — 封装实时推理的缓冲区管理、推理调用、效果器链。

从 RealtimeEngine 中拆分出的推理调度组件，负责：
- 处理状态初始化（采样率/块大小/缓存/重采样/效果器）
- 输入准备（单声道转换、降噪、缓存轮换）
- 推理调用（InferencePipeline.infer + 模型→设备重采样）
- 输出处理（RMS 混合、SOLA 对齐）
- 预热推理（CUDA Graph 捕获）
- 缓冲区重置

RealtimeEngine 保留对外接口，内部委托给本组件处理推理逻辑。
"""
from rvc.audio.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
import logging
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample as TatResample

from rvc.audio.effects import AudioProcessor
from rvc.audio.output_router import route_secondary_output, write_main_output

logger = logging.getLogger(__name__)


class InferenceRunner:
    """推理运行器 — 管理实时推理的缓冲区与处理流程。"""

    def __init__(self, pipeline, runtime_params, device: str, function: str = "vc"):
        """
        Args:
            pipeline: InferencePipeline 实例（已加载模型）
            runtime_params: InferenceConfig 实例（推理参数，运行中可实时修改）
            device: 计算设备（如 "cuda:0"）
            function: 处理模式（"vc" 变声 / "mono" 直通）
        """
        self.pipeline = pipeline
        self.runtime_params = runtime_params
        self._device = device
        self.function = function

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

        # 缓冲区
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

    def init_processing(self, sr: int, block_t: float, cf_t: float, extra_t: float,
                        channels: int, sr_model: int) -> None:
        """设备无关的处理状态初始化（采样率/块大小/缓存/重采样/降噪）。

        实时 setup() 与离线 process_file() 共用，保证两条路径算法完全一致。

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

        self.block_samples = int(np.round(block_t * sr / zc)) * zc
        self.crossfade_samples = int(np.round(cf_t * sr / zc)) * zc
        self.sola_buffer_samples = min(self.crossfade_samples, 4 * zc)
        self.sola_search_samples = zc
        self.extra_samples = int(np.round(extra_t * sr / zc)) * zc

        self.block_samples_16k = HUBERT_FRAME_SIZE * self.block_samples // zc
        self.skip_head = self.extra_samples // zc
        self.return_length = (self.block_samples + self.sola_buffer_samples + self.sola_search_samples) // zc

        n = self.extra_samples + self.crossfade_samples + self.sola_search_samples + self.block_samples
        self.input_wav = torch.zeros(n, device=self._device)
        self.input_wav_res = torch.zeros(HUBERT_FRAME_SIZE * n // zc, device=self._device)
        self.input_wav_work = torch.empty_like(self.input_wav)
        self.input_wav_res_work = torch.empty_like(self.input_wav_res)

        # 输入侧 pinned buffer（CPU↔GPU 非阻塞拷贝复用）
        self._in_pin = torch.empty(self.block_samples, dtype=torch.float32, pin_memory=True)

        # 重采样器
        self.resampler = TatResample(sr, HUBERT_SAMPLE_RATE, dtype=torch.float32).to(self._device)
        if sr_model != sr:
            self.resampler_model2dev = TatResample(sr_model, sr, dtype=torch.float32).to(self._device)
        else:
            self.resampler_model2dev = None

        # 效果器（降噪 / RMS / SOLA）
        self.processor.setup(
            sr, self.block_samples, self.crossfade_samples,
            self.sola_search_samples, self._device,
        )

    def warmup(self, n: int = 2) -> None:
        """开流前用静音数据跑 n 次完整回调，完成 CUDA Graph 捕获。

        需在 init_processing 之后调用。失败只警告，不影响运行。
        """
        if self.pipeline is None or self.input_wav_res is None:
            return
        frames = self.block_samples
        indata = np.zeros((frames, self.channels), dtype=np.float32)
        outdata = np.zeros((frames, self.channels), dtype=np.float32)
        with torch.no_grad():
            for _ in range(n):
                self.process_block(indata, outdata, frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_buffers(self) -> None:
        """重置所有被预热污染的缓冲区（pitch 缓存/效果器/输入缓存）。"""
        if self.pipeline is not None:
            self.pipeline.reset_pitch_cache()
        self.processor.reset()
        if self.input_wav is not None:
            self.input_wav.zero_()
        if self.input_wav_res is not None:
            self.input_wav_res.zero_()

    def process_block(self, indata: np.ndarray, outdata: np.ndarray, frames: int) -> None:
        """处理一个音频块（实时回调主函数）。

        编排各处理阶段：输入准备 → 缓存轮换/降采样 → 推理 → RMS → SOLA → 输出

        Args:
            indata: 输入音频 (frames, channels) float32
            outdata: 输出音频缓冲区 (frames, channels) float32
            frames: 样本帧数
        """
        t0 = time.perf_counter()
        params = self.runtime_params

        # 快照本回调内多次使用的参数
        p_rms_mix = params.rms_mix

        with torch.no_grad():
            # 阶段1-2: 输入准备 + 降噪 + 缓存轮换
            mono = self._prepare_input(indata)
            n = mono.shape[0]
            self._in_pin[:n].copy_(torch.from_numpy(mono), non_blocking=True)
            mono = self._in_pin[:n].to(self._device, non_blocking=True)
            self._update_input_buffers(mono)

            # 阶段3: 语音转换推理
            infer = self._run_inference()

            # 阶段4-5: RMS 混合 + SOLA 对齐
            ref = self.input_wav[self.extra_samples:]
            chunk = self.processor.process_output(infer, ref, p_rms_mix, self.function == "vc")

            # 阶段6: 输出写入（主输出 + 副输出路由由调用方处理）
            write_main_output(chunk, outdata, self.channels)

        self.infer_ms = (time.perf_counter() - t0) * 1000

    def route_secondary_output(self, outdata: np.ndarray, stream2, out2_q: queue.Queue,
                                enable_out2: bool) -> None:
        """副输出路由（从 process_block 分离，由 RealtimeEngine 调用）。"""
        route_secondary_output(outdata, stream2, out2_q, enable_out2)

    def _prepare_input(self, indata: np.ndarray) -> np.ndarray:
        """将输入的立体声/单声道转换为处理的单声道信号。"""
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        return np.ascontiguousarray(mono)

    def _update_input_buffers(self, mono: torch.Tensor) -> None:
        """轮换输入缓存并执行降采样到16kHz（mono 为 GPU tensor）。"""
        # 输入wav缓冲区轮换
        self.input_wav_work[:-self.block_samples].copy_(self.input_wav[self.block_samples:])
        self.input_wav_work[-self.block_samples:].zero_()
        self.input_wav, self.input_wav_work = self.input_wav_work, self.input_wav
        self.input_wav[-mono.shape[0]:] = mono

        # 降采样输入到16kHz供HuBERT特征提取使用
        self.input_wav_res_work[:-self.block_samples_16k].copy_(self.input_wav_res[self.block_samples_16k:])
        self.input_wav_res_work[-self.block_samples_16k:].zero_()
        self.input_wav_res, self.input_wav_res_work = self.input_wav_res_work, self.input_wav_res
        # 取额外 2*hz_centis 上下文喂 resampler（抵消重采样延迟），输出跳过前 HUBERT_FRAME_SIZE 样本
        resampler_in = self.input_wav[-mono.shape[0] - 2 * self.hz_centis:]
        resampler_out = self.resampler(resampler_in)[HUBERT_FRAME_SIZE:]
        target_len = HUBERT_FRAME_SIZE * (mono.shape[0] // self.hz_centis + 1)
        self.input_wav_res[-target_len:] = resampler_out

    def _run_inference(self) -> torch.Tensor:
        """执行语音转换推理或直通模式。

        推理返回长度可能因 formant 重采样 / model2dev 重采样有少量偏差，
        这里统一修正到期望长度（截断或补零），避免下游 SOLA 越界。
        """
        if self.function == "vc" and self.pipeline:
            infer = self.pipeline.infer(
                self.input_wav_res, self.runtime_params,
                self.block_samples_16k, self.skip_head, self.return_length,
            )
            if self.resampler_model2dev:
                infer = self.resampler_model2dev(infer)
        else:
            infer = self.input_wav[self.extra_samples:].clone()
        # 修正长度到期望大小
        expected = self.block_samples + self.sola_buffer_samples + self.sola_search_samples
        if infer.shape[0] > expected:
            infer = infer[:expected]
        elif infer.shape[0] < expected:
            infer = F.pad(infer, (0, expected - infer.shape[0]))
        return infer
