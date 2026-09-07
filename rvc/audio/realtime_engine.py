"""实时音频引擎 — 管理 sounddevice 流、缓冲区、SOLA、声学效果。

架构重构后：RealtimeEngine 作为门面（Facade），内部委托给子组件：
- AudioStreamManager: 设备/流管理（PortAudio 封装）
- InferenceRunner: 推理调度（缓冲区/推理/效果器）
- ModelSessionManager: 模型生命周期（通过 VCPipeline 间接使用）

对外接口（start/stop/load_model/process_file）保持不变，GUI 层无需修改。
"""
import logging
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from rvc.audio.inference_runner import InferenceRunner
from rvc.audio.stream_manager import AudioStreamManager
from rvc.core.config import InferenceConfig
from rvc.runtime import Config

logger = logging.getLogger(__name__)

# 实时推理错误容忍配置
MAX_CONSECUTIVE_ERRORS = 3  # 连续错误达到此阈值后停止推理


class RealtimeEngine:
    """实时音频引擎 — 门面类，内部委托给 AudioStreamManager 和 InferenceRunner。"""

    def __init__(self, runtime_params, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params
        self.inference_cache = inference_cache
        self.on_runtime_error = on_runtime_error
        self.pipeline = None
        self.running = False
        self.function = "vc"

        # 子组件
        self._stream_mgr = AudioStreamManager()
        self._runner = None  # InferenceRunner（setup 时创建）

        # 错误统计
        self.error_count = 0
        self.max_error_count = MAX_CONSECUTIVE_ERRORS
        self.last_error = ""
        self.runtime_error_pending = False

        # 性能统计
        self.infer_ms = 0.0
        self.measure_ms = 0.0  # 硬件时间戳实测端到端延迟（EMA 平滑）

        self._cfg = Config()  # 单例缓存，避免多处重复获取
        self.pth_path = ""

    # ── 模型加载 ──

    def load_model(self, pth, force=False, hubert="chinese"):
        """加载模型（创建 VCPipeline）。

        切换模型时清除 f0 提取器的旧 CUDA Graph 缓存。
        """
        if self.inference_cache:
            self.inference_cache.clear_f0_cuda_graph_caches()
        if not force and self.pipeline and self.pth_path == pth:
            return self.pipeline.target_sr
        from rvc.inference.pipeline import VCPipeline
        try:
            self.pipeline = VCPipeline(self._cfg, pth, self.inference_cache, hubert=hubert)
            self.pipeline.load()
            self.pth_path = pth
            return self.pipeline.target_sr
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.pipeline = None
            raise

    # ── 引擎启动/停止 ──

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t, out2_dev_idx=None):
        """启动实时变声引擎。"""
        if self._stream_mgr.stream is not None:
            self.stop()
        self.error_count = 0
        self.last_error = ""
        self.runtime_error_pending = False

        sd.default.device = [in_dev, out_dev]
        sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        sr_model = self.pipeline.target_sr
        sr = sr_model if sr_type == "sr_model" else sr_dev

        # 设备校验与日志
        self._stream_mgr.validate_and_log_devices(in_dev, out_dev, out2_dev_idx)
        channels = self._stream_mgr.channels

        # 推理运行器初始化
        self._runner = InferenceRunner(self.pipeline, self.runtime_params, self._cfg.device, self.function)
        self._runner.init_processing(sr, block_t, cf_t, extra_t, channels, sr_model)

        # 预热 + 重置缓冲区
        self._runner.warmup(2)
        self._runner.reset_buffers()

        # 启动主流
        self._stream_mgr.start_main_stream(self._cb, sr, channels, self._runner.block_samples)
        self.running = True

        # 启动副输出（如果指定）
        if out2_dev_idx is not None:
            self.setup_out2(out2_dev_idx)

    def setup_out2(self, dev_idx):
        """启动副输出流。"""
        if self._runner is None:
            raise RuntimeError("请先启动主引擎再设置副输出")
        self._stream_mgr.start_secondary_output(
            dev_idx, self._runner.sr, self._runner.channels, self._runner.block_samples
        )

    def stop(self):
        """停止引擎，关闭所有流。"""
        self.running = False
        self.error_count = 0
        self.runtime_error_pending = False
        self._stream_mgr.stop_all()
        # 等待 GPU 上所有推理操作完成，确保快速 stop→start 时旧 kernel 已结束。
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ── 音频回调 ──

    def _cb(self, indata, outdata, frames, times, status):
        """sounddevice 回调函数 — 委托给 InferenceRunner.process_block。"""
        try:
            # 硬件时间戳实测端到端延迟
            d = float(times.outputBufferDacTime - times.inputBufferAdcTime)
            if 0 < d < 2:
                ms = d * 1000
                self.measure_ms = ms if self.measure_ms <= 0 else self.measure_ms * 0.7 + ms * 0.3

            # 委托给推理运行器
            self._runner.process_block(indata, outdata, frames)
            self.infer_ms = self._runner.infer_ms

            # 副输出路由
            if self._stream_mgr.enable_out2:
                self._stream_mgr.route_secondary_output(
                    outdata, self._stream_mgr.stream2, self._stream_mgr.out2_q, True
                )

            self.error_count = 0
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error("音频回调异常(%d/%d): %s", self.error_count, self.max_error_count, e, exc_info=True)
            outdata[:] = 0
            if self.error_count >= self.max_error_count and not self.runtime_error_pending:
                self.running = False
                self.runtime_error_pending = True
                if self.on_runtime_error:
                    self.on_runtime_error(self.last_error or "实时推理失败")
                raise sd.CallbackStop

    # ── 离线文件推理 ──

    def process_file(self, task, *, block_t=0.25, cf_t=0.05, extra_t=2.5,
                     pad_sec=3.0, progress_cb=None):
        """离线文件流式推理：「模拟播放→转换→写录」。

        把整段音频当作持续输入流，逐块走实时 process_block（降噪/RMS/SOLA/缓存轮换），
        与实时完全同一算法。显存封顶，音质 = 实时音质。
        """
        sr_model = self.pipeline.target_sr
        tgt_sr = sr_model
        wav = self._load_audio_at_sr(task.input_path, tgt_sr)

        self.runtime_params = task
        if self.pipeline:
            self.pipeline.reset_pitch_cache()
        self.function = "vc"

        # 创建推理运行器（不启动音频流）
        self._runner = InferenceRunner(self.pipeline, self.runtime_params, self._cfg.device, self.function)
        self._runner.init_processing(tgt_sr, block_t, cf_t, extra_t, channels=1, sr_model=sr_model)
        self._runner.warmup(2)

        result = self._infer_stream(wav, self._runner.block_samples, int(tgt_sr * pad_sec), progress_cb)
        self._write_output_wav(result, task.output_path, tgt_sr)
        return result

    def _load_audio_at_sr(self, input_path, tgt_sr):
        """加载音频并重采样到目标采样率（float32, 单声道）。"""
        from rvc.audio.loader import load_audio
        wav, _ = load_audio(input_path, tgt_sr)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _infer_stream(self, wav, block, pad, progress_cb):
        """把整段音频按块走实时 process_block，返回裁剪掉 pad 的输出。"""
        padded = np.pad(wav, (pad, pad), mode="reflect")
        if len(padded) % block:
            padded = np.concatenate([padded, np.zeros(block - len(padded) % block, dtype=np.float32)])

        total_blocks = len(padded) // block
        out_chunks = []
        for i in range(total_blocks):
            seg = padded[i * block: (i + 1) * block]
            outdata = np.zeros((block, 1), dtype=np.float32)
            self._runner.process_block(seg, outdata, block)
            out_chunks.append(outdata[:, 0])
            if progress_cb:
                progress_cb(i + 1, total_blocks)
        return np.concatenate(out_chunks)[pad: pad + len(wav)]

    def _write_output_wav(self, result, output_path, tgt_sr):
        """峰值归一化（防削波）后写出 wav。"""
        import soundfile as sf

        audio_max = np.abs(result).max() / 0.99
        if audio_max > 1:
            result = result / audio_max
        sf.write(output_path, result, tgt_sr, subtype="FLOAT")

    # ── 兼容属性（供 GUI 层访问）──

    @property
    def stream(self):
        return self._stream_mgr.stream

    @property
    def stream2(self):
        return self._stream_mgr.stream2

    @property
    def out2_q(self):
        return self._stream_mgr.out2_q

    @property
    def enable_out2(self):
        return self._stream_mgr.enable_out2

    @property
    def sr(self):
        return self._runner.sr if self._runner else None

    @property
    def channels(self):
        return self._runner.channels if self._runner else 1

    @property
    def block_samples(self):
        return self._runner.block_samples if self._runner else 0
