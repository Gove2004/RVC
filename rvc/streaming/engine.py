"""变声引擎门面 — 实时与离线共用的唯一入口。

VoiceEngine 作为门面（Facade），内部委托给子组件：

- AudioStreams: 设备/流管理（PortAudio 封装）
- InferenceRunner: 推理调度（缓冲区/推理/效果器）
- ModelSessions: 模型生命周期（通过 InferencePipeline 间接使用）

对外接口：
- 实时: setup() / setup_out2() / stop() / running / measure_ms / infer_ms
- 模型: load_model(pth, force=False, hubert=...)
- 离线: process_file(task, pad_sec=3.0, progress_cb=None)

离线与实时走同一 InferenceRunner.process_block，保证音质一致。
"""

import gc
import logging
import threading

import numpy as np
import sounddevice as sd
import torch

from rvc.streaming.runner import InferenceRunner
from rvc.streaming.stream import AudioStreams
from rvc.core.config import InferenceParams
from rvc.core.errors import CancelledError
from rvc.runtime import RuntimeDeviceConfig

logger = logging.getLogger(__name__)


# 实时启动预热迭代次数（取值与依据见 _create_runner）
_WARMUP_ITERATIONS = 30

# 异步 CUDA 缓存回收间隔（块数）。每 N 块触发一次后台 empty_cache，
# 不阻塞音频回调。实际时间间隔 = N × block_time（默认 0.1s → 约 3s）。
# 参考 Nvidia 优化补丁的核心内存创新。
_CUDA_CLEANUP_EVERY_N = 30


class VoiceEngine:
    """变声引擎门面 — 委托 AudioStreams 管流、InferenceRunner 推理。"""

    def __init__(self, runtime_params: InferenceParams, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params
        self.inference_cache = inference_cache
        self.on_runtime_error = on_runtime_error
        self.pipeline = None
        self.running = False

        # 子组件
        self._streams = AudioStreams()
        self._runner = None  # InferenceRunner（setup 时创建）

        # 性能统计（GUI 读取）
        self.infer_ms = 0.0
        self.measure_ms = 0.0  # 端到端延迟（块时长+推理耗时口径，瞬时值）

        # 错误状态由 InferenceRunner 管理（通过属性代理访问）
        self._cfg = RuntimeDeviceConfig()  # 单例缓存，避免多处重复获取
        self.pth_path = ""
        self.sr_model = 0  # 模型目标采样率（setup 时赋值）
        self.sr_dev = 0  # 输入设备默认采样率（setup 时赋值）

        # 异步 CUDA 缓存回收（后台线程，不阻塞音频回调）
        self._cleanup_counter = 0
        self._cleanup_pending = False
        self._cleanup_lock = threading.Lock()
        self._cleanup_thread = None

    # ── 错误状态属性代理（委托给 InferenceRunner） ──

    @property
    def error_count(self):
        return self._runner.state.error_count if self._runner else 0

    @property
    def last_error(self):
        return self._runner.state.last_error if self._runner else ""

    @property
    def input_pitch(self) -> float:
        """当前输入音高（Hz，音域映射之前的原始值；无声/未运行为 0）。

        只读遥测——GUI 延迟/音高显示经此门面取数，禁止穿透 pipeline 内部状态。
        热路径（pipeline）只留 GPU 定形统计，本属性被 GUI 低频读取时
        才在这里做唯一一次 GPU→CPU 同步取值。
        """
        if self._runner is None:
            return 0.0
        stat = self._runner.state.last_input_pitch_gpu
        if stat is None:
            return 0.0
        pitch_sum, count = stat
        # count==0（本块全清音）→ 0，与本文档「无声为 0」的契约一致
        # （旧实现此时保留上一块旧值，与文档矛盾，P1 重构时一并修正）。
        # clamp(min=1) 防除零，全程单次 .item() 同步。
        return float((pitch_sum / count.clamp(min=1)).item())

    @property
    def runtime_error_pending(self):
        return self._runner.state.runtime_error_pending if self._runner else False

    @runtime_error_pending.setter
    def runtime_error_pending(self, value):
        if self._runner:
            self._runner.state.runtime_error_pending = value

    # ── 模型加载 ──

    def _clear_f0_cuda_graph_caches(self) -> None:
        """无条件清空 F0 提取器的 CUDA Graph 缓存（§11.2：显式步骤，非顺手前置）。

        load_model 入口必经——包括命中缓存快路径与首次加载，
        不依赖任何条件；离线场景 inference_cache=None，但 pipeline 内部
        回退到全局 default_inference_cache，故对「显式 cache 或全局单例」
        统一清理，避免离线切模型时残留旧图。
        """
        from rvc.pipeline.cache import default_inference_cache

        (self.inference_cache or default_inference_cache).clear_f0_cuda_graph_caches()

    def load_model(self, pth, force=False, hubert="chinese"):
        """加载模型（创建 InferencePipeline）。

        入口先无条件清空 f0 提取器的旧 CUDA Graph 缓存
        （见 _clear_f0_cuda_graph_caches）。
        """
        self._clear_f0_cuda_graph_caches()
        if not force and self.pipeline and self.pth_path == pth:
            return self.pipeline.target_sr
        from rvc.pipeline.pipeline import InferencePipeline

        try:
            self.pipeline = InferencePipeline(self._cfg, pth, self.inference_cache, hubert=hubert)
            self.pipeline.load()
            # 模型权重已搬到 GPU，释放加载过程中 CPU 侧的临时张量缓存还给 OS
            torch.cuda.empty_cache()
            gc.collect()
            self.pth_path = pth
            return self.pipeline.target_sr
        except Exception as e:
            logger.error("模型加载失败：%s", e, exc_info=True)
            self.pipeline = None
            raise

    # ── 引擎启动/停止 ──

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t, out2_dev_idx=None):
        """启动实时变声引擎。"""
        if self._streams.stream is not None:
            self.stop()
        sd.default.device = [in_dev, out_dev]
        self.sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        self.sr_model = self.pipeline.target_sr
        sr = self.pipeline.target_sr if sr_type == "sr_model" else self.sr_dev

        # 设备校验与日志
        self._streams.validate_and_log_devices(in_dev, out_dev, out2_dev_idx)
        channels = self._streams.channels

        # 推理运行器初始化（实时模式：预热后重置缓冲区）
        self._create_runner(sr, channels, block_t, cf_t, extra_t, self.pipeline.target_sr, reset_buffers=True)

        # 启动主流（显式指定设备，不依赖 sd.default.device）
        self._streams.start_main_stream(
            self._cb, sr, channels, self._runner.state.block_samples,
            in_dev=in_dev, out_dev=out_dev,
        )
        self.running = True

        # 启动副输出（如果指定）
        if out2_dev_idx is not None:
            self.setup_out2(out2_dev_idx)

    def _create_runner(self, sr, channels, block_t, cf_t, extra_t, sr_model, *, reset_buffers=True):
        """创建并初始化 InferenceRunner（实时 setup 与离线 process_file 共用）。

        Args:
            sr: 工作采样率
            channels: 通道数
            block_t: 块时长（秒）
            cf_t: 交叉淡化时长（秒）
            extra_t: 额外上下文时长（秒）
            sr_model: 模型目标采样率
            reset_buffers: 是否在预热后重置缓冲区（实时需要，离线不需要）
        """
        self._runner = InferenceRunner(self.pipeline, self.runtime_params)
        self._runner.init_processing(sr, block_t, cf_t, extra_t, channels, sr_model)
        self._runner.reset_error_state()
        # 预热 30 次：2 次只够捕获 CUDA Graph，不足以让 GPU 升频/缓存预热。
        # 笔记本 GPU 冷启动时频率低、cuDNN 未调优、L2 缓存为空，
        # 首次实际推理会卡顿；停止重开后 GPU 仍热所以流畅。
        # 30 次约 300-600ms，换来首次启动即流畅。
        self._runner.warmup(_WARMUP_ITERATIONS)
        if reset_buffers:
            self._runner.reset_buffers()

    def setup_out2(self, dev_idx):
        """启动副输出流。"""
        if self._runner is None:
            raise RuntimeError("请先启动主引擎再设置副输出")
        self._streams.start_secondary_output(
            dev_idx, self._runner.state.work_sr, self._runner.state.channels, self._runner.state.block_samples
        )

    def _maybe_async_cleanup(self) -> None:
        """每 _CUDA_CLEANUP_EVERY_N 块触发一次后台线程 empty_cache。

        empty_cache() 会阻塞 GPU，放在后台线程执行不影响音频回调。
        用 lock + pending 标志防重叠：上一次回收未完成时跳过本次。
        """
        if not torch.cuda.is_available():
            return
        self._cleanup_counter += 1
        if self._cleanup_counter < _CUDA_CLEANUP_EVERY_N:
            return
        self._cleanup_counter = 0
        with self._cleanup_lock:
            if self._cleanup_pending:
                return
            self._cleanup_pending = True

        def _cleanup():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.warning("异步 CUDA 缓存回收失败", exc_info=True)
            finally:
                self._cleanup_pending = False

        self._cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
        self._cleanup_thread.start()

    def stop(self):
        """停止引擎，关闭所有流。"""
        self.running = False
        self._cleanup_counter = 0
        # 等待后台回收线程结束（最多 1s），避免与下方同步 empty_cache 并发
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
        self._cleanup_thread = None
        if self._runner:
            self._runner.reset_error_state()
        self._streams.stop_all()
        # 等待 GPU 上所有推理操作完成，确保快速 stop→start 时旧 kernel 已结束。
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # 显式回收：gc 释放 Python 侧引用（如临时张量/效果器缓存），
        # empty_cache 把 PyTorch 缓存池中的空闲块还给 CUDA 驱动。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 音频回调 ──

    def _cb(self, indata, outdata, frames, times, status):
        """sounddevice 回调函数 — 委托给 InferenceRunner.process_block。"""
        try:
            # 端到端延迟 = 块时长 + 本块推理耗时（块对齐后相位抵消，恒定项）。
            # 不用 PortAudio 硬件时间戳（outputBufferDacTime - inputBufferAdcTime）：
            # WASAPI 共享模式下它不含真实缓冲深度（输入捕获环 + 输出渲染环各约
            # 一个块），实测比麦克风对拍小 ~180ms。
            # 残差：两端端点 APO/驱动缓冲约 10-30ms，恒定、不在此显示。
            self._runner.process_block(indata, outdata, frames)
            self.infer_ms = self._runner.state.infer_ms
            state = self._runner.state
            if state.work_sr:
                self.measure_ms = state.block_samples / state.work_sr * 1000 + self.infer_ms

            # 副输出路由
            if self._streams.enable_out2:
                self._runner.route_secondary_output(
                    outdata, self._streams.stream2, self._streams.out2_q,
                )

            self._runner.acknowledge_success()
            self._maybe_async_cleanup()
        except Exception as e:
            should_stop = self._runner.handle_error(e)
            logger.error("音频回调异常(%d/%d)：%s", self._runner.state.error_count, self._runner.state.max_error_count, e, exc_info=True)

            outdata[:] = 0
            if should_stop:
                self.running = False
                if self.on_runtime_error:
                    self.on_runtime_error(self._runner.state.last_error or "实时推理失败")
                raise sd.CallbackStop

    # ── 离线文件推理 ──

    def process_file(self, task, *, pad_sec=3.0, progress_cb=None, cancel_cb=None):
        """离线文件流式推理：「模拟播放→转换→写录」。

        把整段音频当作持续输入流，逐块走实时 process_block（RMS/SOLA/缓存轮换），
        与实时完全同一算法。显存封顶，音质 = 实时音质。
        块/交叉淡化/额外上下文参数从 task（InferenceParams）中读取，与实时一致。

        调用契约（现状语义，用户裁定保持）：离线与实时共用同一 engine/runner，
        本方法会替换 self._runner 并原地更新 runtime_params——
        因此**实时运行中不得调用**；先 stop() 再离线转换。

        Args:
            cancel_cb: 可选回调，返回 True 时中止推理（抛出 CancelledError）。
        """
        sr_model = self.pipeline.target_sr
        tgt_sr = sr_model
        wav = self._load_audio_at_sr(task.input_path, tgt_sr)

        # 原地更新 runtime_params 字段（不替换对象引用，已创建的 runner 自动生效）
        self.runtime_params.update_from(task)

        # 从 task 中读取缓冲区参数，与实时 setup 保持一致
        block_t = task.buffer.block_time
        cf_t = task.buffer.crossfade_time
        extra_t = task.buffer.extra_time

        # 创建推理运行器（离线模式：不重置pipeline缓冲区，避免清除pad上下文）
        self._create_runner(tgt_sr, 1, block_t, cf_t, extra_t, sr_model, reset_buffers=False)
        # 但必须重置有状态效果器（airflow等），避免warmup零块污染上下文缓存
        self._runner.effects.reset()

        result = self._infer_stream(wav, self._runner.state.block_samples, int(tgt_sr * pad_sec), progress_cb, cancel_cb)
        self._write_output_wav(result, task.output_path, tgt_sr)
        return result

    def _load_audio_at_sr(self, input_path, tgt_sr):
        """加载音频并重采样到目标采样率（float32, 单声道）。"""
        from rvc.io.audio_file import load_audio

        wav, _ = load_audio(input_path, tgt_sr)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _infer_stream(self, wav, block, pad, progress_cb, cancel_cb=None):
        """把整段音频按块走实时 process_block，返回裁剪掉 pad 的输出。

        Args:
            cancel_cb: 可选回调，返回 True 时抛出 CancelledError 中止推理。
        """
        padded = np.pad(wav, (pad, pad), mode="reflect")
        if len(padded) % block:
            padded = np.concatenate([padded, np.zeros(block - len(padded) % block, dtype=np.float32)])

        total_blocks = len(padded) // block
        out_chunks = []
        for i in range(total_blocks):
            if cancel_cb and cancel_cb():
                raise CancelledError("离线推理已取消")
            seg = padded[i * block: (i + 1) * block]
            outdata = np.zeros((block, 1), dtype=np.float32)
            self._runner.process_block(seg, outdata, block)
            out_chunks.append(outdata[:, 0])
            if progress_cb:
                progress_cb(i + 1, total_blocks)
        return np.concatenate(out_chunks)[pad: pad + len(wav)]

    def _write_output_wav(self, result, output_path, tgt_sr):
        """峰值归一化（防削波）后写出 wav。"""
        from rvc.io.wav_file import write_wav

        if len(result) == 0:
            write_wav(output_path, result, tgt_sr, subtype="FLOAT")
            return

        audio_max = np.abs(result).max() / 0.99
        if audio_max > 1:
            result = result / audio_max
        write_wav(output_path, result, tgt_sr, subtype="FLOAT")

