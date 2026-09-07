"""实时音频引擎 — 管理 sounddevice 流、缓冲区、SOLA、声学效果。

架构: RealtimeEngine 拥有一个 VCPipeline 实例，在 sounddevice 回调中驱动推理。
audio 层依赖 inference 层是有意为之——回调必须协调流计时与模型推理。
"""
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from torchaudio.transforms import Resample as TatResample

from rvc.audio.effects import AudioProcessor
from rvc.audio.output_router import route_secondary_output, write_main_output
from rvc.config import InferenceConfig
from rvc.inference.runner import InferenceRunner
from rvc.runtime import Config

logger = logging.getLogger(__name__)

# 实时推理错误容忍配置
MAX_CONSECUTIVE_ERRORS = 3  # 连续错误达到此阈值后停止推理


class RealtimeEngine:
    def __init__(self, runtime_params, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params
        self.inference_cache = inference_cache
        self.on_runtime_error = on_runtime_error
        self.pipeline = None
        self.runner = None  # InferenceRunner（load_model 后创建）
        self.stream = None
        self.stream2 = None
        self.running = False
        self.function = "vc"
        self.out2_q = queue.Queue(maxsize=10)
        self.enable_out2 = False  # 副输出开关（setup_out2 时设 True，stop 时重置；不依赖 InferenceConfig）

        # ── 处理状态（setup()/process_file() 经 _init_processing 填充）──
        self.sr = None; self.sr_dev = None; self.sr_model = None
        self.hz_centis = None; self.channels = 1
        self.block_samples = 0; self.block_samples_16k = 0
        self.crossfade_samples = 0; self.sola_buffer_samples = 0
        self.sola_search_samples = 0; self.extra_samples = 0
        self.skip_head = 0; self.return_length = 0

        self.input_wav = None; self.input_wav_res = None
        self.input_wav_work = None; self.input_wav_res_work = None
        self.sola_buffer = None
        self.fade_in = None; self.fade_out = None; self.sola_norm_kernel = None
        self.resampler = None; self.resampler_model2dev = None
        self._in_pin = None          # 输入侧 pinned buffer（CPU↔GPU 非阻塞拷贝复用）
        self.processor = AudioProcessor()

        self.pth_path = ""
        self.infer_ms = 0.0
        self.measure_ms = 0.0  # 硬件时间戳实测端到端延迟（EMA 平滑）
        self.error_count = 0
        self.max_error_count = MAX_CONSECUTIVE_ERRORS
        self.last_error = ""
        self.runtime_error_pending = False

    def load_model(self, pth, force=False, hubert="chinese"):
        if not force and self.pipeline and self.pth_path == pth:
            return self.pipeline.target_sr
        from rvc.inference.pipeline import VCPipeline
        config = Config()  # load_model 在 _init_processing 之前调用，需单独获取 Config
        # 诊断：清除所有模型缓存，每次都重新加载。
        # 若清除后快速重启不再沙哑，说明根因在模型缓存层。
        self.inference_cache.clear()
        try:
            self.pipeline = VCPipeline(config, pth, self.inference_cache, hubert=hubert)
            self.pipeline.load()
            self.runner = InferenceRunner(self.pipeline)
            self.pth_path = pth
            return self.pipeline.target_sr
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.pipeline = None
            self.runner = None
            raise

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t, out2_dev_idx=None):
        if self.stream is not None:
            self.stop()
        self.error_count = 0
        self.last_error = ""
        self.runtime_error_pending = False
        sd.default.device = [in_dev, out_dev]
        self.sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        self.sr_model = self.pipeline.target_sr
        self.sr = self.sr_model if sr_type == "sr_model" else self.sr_dev

        in_info, out_info = sd.query_devices(in_dev), sd.query_devices(out_dev)
        in_max = int(in_info["max_input_channels"])
        out_max = int(out_info["max_output_channels"])
        if in_max <= 0:
            raise RuntimeError(
                f"输入设备不支持录音（max_input_channels={in_max}）："
                f"索引 {in_dev}「{in_info.get('name', '?')}」。请在设备设置中选择支持输入的设备。"
            )
        if out_max <= 0:
            raise RuntimeError(
                f"输出设备不支持播放（max_output_channels={out_max}）："
                f"索引 {out_dev}「{out_info.get('name', '?')}」。请在设备设置中选择支持输出的设备。"
            )
        self.channels = min(in_max, out_max, 2)
        logger.info("音频设备：")
        logger.info("  · 麦克风：%s", in_info.get('name', '?'))
        logger.info("  · 主输出：%s [%dch]", out_info.get('name', '?'), self.channels)
        if out2_dev_idx is not None:
            out2_info = sd.query_devices(out2_dev_idx)
            logger.info("  · 副输出：%s", out2_info.get('name', f"#{out2_dev_idx}"))

        self._init_processing(self.sr, block_t, cf_t, extra_t, self.channels)

        # 重置 pitch 缓存：多次 stop/setup 后旧 pitch 数据会污染新会话，导致声音沙哑/失真
        if self.runner is not None:
            self.runner.reset()

        # 开流前预热推理：首次推理会触发 CUDA Graph 捕获（每模型 3 次 warmup 前向 + capture，
        # 单块可能数百 ms），提前用静音数据跑完，让首次真实回调即为热状态。
        # 这就是「停止后重新开始延迟变低」的原因——图已捕获；现在把它提前到开流前。
        self.warmup_inference(2)

        # warmup 用静音数据跑推理会污染所有缓冲区（pitch/sola/输入缓存），
        # 静音段 f0 提取可能输出随机值，SOLA 缓冲区会残留静音段的交叉淡化状态。
        # 必须在 warmup 后彻底重置所有运行时缓冲区，确保首次真实推理从干净状态开始。
        if self.runner is not None:
            self.runner.reset()
        self.sola_buffer.zero_()
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        if self.nr_ss is not None:
            self.nr_ss.reset()

        self.stream = sd.Stream(callback=self._cb, blocksize=self.block_samples, samplerate=self.sr, channels=self.channels, dtype="float32")
        self.stream.start()
        self.running = True

    def _init_processing(self, sr, block_t, cf_t, extra_t, channels):
        """设备无关的处理状态初始化（采样率/块大小/缓存/重采样/降噪）。

        setup() 与离线文件流式推理 process_file() 共用，保证两条路径算法完全一致。
        需先设置 self.sr_model（= pipeline.target_sr），用于判断是否需模型→设备重采样。
        """
        cfg = Config()  # 单例，函数内获取避免模块导入时触发 CUDA 探测
        self._device = cfg.device
        self.sr = sr
        self.channels = channels
        zc = self.sr // 100
        self.block_samples = int(np.round(block_t * self.sr / zc)) * zc
        self.crossfade_samples = int(np.round(cf_t * self.sr / zc)) * zc
        self.sola_buffer_samples = min(self.crossfade_samples, 4 * zc)
        self.sola_search_samples = zc
        self.extra_samples = int(np.round(extra_t * self.sr / zc)) * zc

        self.block_samples_16k = 160 * self.block_samples // zc
        self.skip_head = self.extra_samples // zc
        self.return_length = (self.block_samples + self.sola_buffer_samples + self.sola_search_samples) // zc

        n = self.extra_samples + self.crossfade_samples + self.sola_search_samples + self.block_samples
        self.input_wav = torch.zeros(n, device=cfg.device)
        self.input_wav_res = torch.zeros(160 * n // zc, device=cfg.device)
        self.input_wav_work = torch.empty_like(self.input_wav)
        self.input_wav_res_work = torch.empty_like(self.input_wav_res)
        self.hz_centis = zc
        # 输入侧 pinned buffer（CPU↔GPU 非阻塞拷贝复用，避免每块分配）
        self._in_pin = torch.empty(self.block_samples, dtype=torch.float32, pin_memory=True)

        self.resampler = TatResample(self.sr, 16000, dtype=torch.float32).to(cfg.device)
        if self.sr_model != self.sr:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr, dtype=torch.float32).to(cfg.device)
        else:
            self.resampler_model2dev = None

        # 效果器（降噪 / RMS / SOLA）
        self.processor.setup(
            self.sr, self.block_samples, self.crossfade_samples,
            self.sola_search_samples, cfg.device,
        )

    def warmup_inference(self, n: int = 2):
        """开流前用静音数据跑 n 次完整回调，完成 CUDA Graph 捕获以及
        降噪/重采样/SOLA/输出等所有首次开销，让首次真实回调即为热状态。

        需在 setup() 分配 buffer 之后、开流之前调用（形状与真实回调一致）。
        失败只警告，不影响运行。
        """
        if self.pipeline is None or self.input_wav_res is None:
            return
        self.function = "vc"
        frames = self.block_samples
        indata = np.zeros((frames, self.channels), dtype=np.float32)
        outdata = np.zeros((frames, self.channels), dtype=np.float32)
        with torch.no_grad():
            for _ in range(n):
                self._cb_impl(indata, outdata, frames, None, None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def setup_out2(self, dev_idx):
        def out2_callback(outdata, frames, time_info, status):
            if not self.out2_q.empty():
                data = self.out2_q.get_nowait()
                outdata[:] = data[:frames]
            else:
                outdata[:] = 0
        out2_info = sd.query_devices(dev_idx)
        out2_max = int(out2_info["max_output_channels"])
        if out2_max < self.channels:
            raise RuntimeError(
                f"副输出设备「{out2_info.get('name', '?')}」只支持 {out2_max} 通道，"
                f"但主输出使用 {self.channels} 通道。请选择支持至少 {self.channels} 通道的副输出设备。"
            )
        self.stream2 = sd.OutputStream(
            device=dev_idx, samplerate=self.sr, channels=self.channels,
            dtype="float32", blocksize=self.block_samples, callback=out2_callback
        )
        self.stream2.start()
        self.enable_out2 = True
        logger.debug("副输出流已启动: sr=%d, ch=%d, block=%d", self.sr, self.channels, self.block_samples)
        while not self.out2_q.empty():
            self.out2_q.get_nowait()



    def stop(self):
        self.running = False
        self.error_count = 0
        self.runtime_error_pending = False
        for s in (self.stream2, self.stream):
            if s:
                try:
                    s.abort()
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
        self.stream = self.stream2 = None
        self.enable_out2 = False
        # 等待 GPU 上所有推理操作完成，确保快速 stop→start 时旧 kernel 已结束。
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # PortAudio 流关闭是异步的：abort/close 返回后底层流可能还在释放中。
        # 快速 stop→start 时新流复用旧流资源会导致音频数据混乱（声音沙哑）。
        # 短暂等待确保底层流完全释放。间隔一段时间重启不沙哑正是因为等够了。
        time.sleep(0.15)

    def process_file(self, input_path, output_path, *, params=None,
                     block_t=0.25, cf_t=0.05, extra_t=2.5,
                     f0method=None, protect=None, pad_sec=3.0,
                     progress_cb=None):
        """离线文件流式推理：「模拟播放→转换→写录」。

        把整段音频当作持续输入流，逐块走实时 `_cb_impl`（降噪/RMS/SOLA/缓存轮换），
        与实时完全同一算法。显存封顶（~400MB，不再随音频长度线性增长），音质 = 实时音质。
        输出采样率 = 模型 target_sr。

        Args:
            input_path: 输入音频路径（任意格式）
            output_path: 输出 wav 路径
            params: 运行时参数（rvc.inference.params.Params），缺省用默认
            block_t/cf_t/extra_t: 块时长/交叉淡化/上下文（秒）
            f0method/protect: 可覆盖 params 对应的推理参数
            pad_sec: 前后上下文 pad（秒），保证首尾块上下文充足
            progress_cb: 可选回调 (completed_blocks, total_blocks)

        Returns:
            输出 wav 完整数组 (float32, target_sr 采样率)
        """
        self.sr_model = self.pipeline.target_sr
        tgt_sr = self.sr_model
        wav = self._load_audio_at_sr(input_path, tgt_sr)

        if params is None:
            params = InferenceConfig()
        self.runtime_params = params
        if self.runner:
            self.runner.reset()  # 重置 pitch 缓存，避免跨文件污染
        if f0method is not None:
            self.runtime_params.f0_method = f0method
        if protect is not None:
            self.runtime_params.protect = protect
        self.function = "vc"

        self._init_processing(tgt_sr, block_t, cf_t, extra_t, channels=1)
        self.warmup_inference(2)

        result = self._infer_stream(wav, self.block_samples, int(tgt_sr * pad_sec), progress_cb)
        self._write_output_wav(result, output_path, tgt_sr)
        return result

    def _load_audio_at_sr(self, input_path, tgt_sr):
        """加载音频并重采样到目标采样率（float32, 单声道）。"""
        from rvc.audio.loader import load_audio
        wav, _ = load_audio(input_path, tgt_sr)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _infer_stream(self, wav, block, pad, progress_cb):
        """把整段音频按块走实时 `_cb_impl`，返回裁剪掉 pad 的输出。

        前后补 pad（reflect）、末尾补零到块整数倍，逐块推理后裁掉 pad。
        """
        import numpy as np

        padded = np.pad(wav, (pad, pad), mode="reflect")
        if len(padded) % block:
            padded = np.concatenate([padded, np.zeros(block - len(padded) % block, dtype=np.float32)])

        total_blocks = len(padded) // block
        out_chunks = []
        for i in range(total_blocks):
            seg = padded[i * block: (i + 1) * block]
            outdata = np.zeros((block, self.channels), dtype=np.float32)
            self._cb_impl(seg, outdata, block, None, None)
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

    def _cb(self, indata, outdata, frames, times, status):
        try:
            # 硬件时间戳实测端到端延迟：本块输出被 DAC 播放的时刻 - 本块输入被 ADC 采集的时刻。
            # 这是 PortAudio 声卡时钟域的精确值，包含设备缓冲/攒块/处理全链路。
            d = float(times.outputBufferDacTime - times.inputBufferAdcTime)
            if 0 < d < 2:  # 过滤异常值（时钟跳变/首块）
                ms = d * 1000
                # EMA 平滑：瞬时值每块跳动（块长 150ms 级），显示会乱跳显得不准。
                # 首块直接赋值，之后 0.7/0.3 平滑收敛。
                self.measure_ms = ms if self.measure_ms <= 0 else self.measure_ms * 0.7 + ms * 0.3
            self._cb_impl(indata, outdata, frames, times, status)
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
                raise sd.CallbackStop  # 安全终止流，避免僵尸流继续占用设备导致下次无法重载

    def _cb_impl(self, indata, outdata, frames, times, status):
        """实时音频回调主函数 — 编排各处理阶段：

        输入准备+降噪 → 缓存轮换/降采样 → 推理 → RMS → SOLA → 输出
        """
        t0 = time.perf_counter()
        params = self.runtime_params

        # 快照本回调内多次使用的参数（推理相关参数由 _run_inference 直接读 runtime_params）
        p_rms_mix = params.rms_mix
        p_enable_out2 = self.enable_out2  # 引擎自身状态，不依赖 InferenceConfig
        p_nr_enable = params.denoise.enable
        p_nr_strength = params.denoise.strength

        with torch.no_grad():
            # ── 阶段1-2: 输入准备 + 降噪 + 缓存轮换 ─────────────────
            mono = self._prepare_input(indata)
            # 输入一次性上 GPU（降噪/推理/输出共用，避免 CPU↔GPU 往返）。
            # 预分配 pinned buffer + 非阻塞拷贝，避免每块重复分配临时张量。
            src = torch.from_numpy(mono)
            self._in_pin[: src.shape[0]].copy_(src, non_blocking=True)
            mono = self._in_pin[: src.shape[0]].to(self._device, non_blocking=True)
            mono = self.processor.process_input(mono, p_nr_enable, p_nr_strength)
            self._update_input_buffers(mono)

            # ── 阶段3: 语音转换推理 ───────────────────────────────────
            infer = self._run_inference()

            # ── 阶段4-5: RMS 混合 + SOLA 对齐 ────────────────────────
            ref = self.input_wav[self.extra_samples:]
            chunk = self.processor.process_output(infer, ref, p_rms_mix, self.function == "vc")

            # ── 阶段6: 输出写入与副输出路由 ───────────────────────────
            self._write_output(chunk, outdata, p_enable_out2)

        self.infer_ms = (time.perf_counter() - t0) * 1000


    def _write_output(self, chunk, outdata, enable_out2):
        """主输出写入 + 副输出路由"""
        write_main_output(chunk, outdata, self.channels)
        route_secondary_output(outdata, self.stream2, self.out2_q, enable_out2)

    def _prepare_input(self, indata):
        """将输入的立体声/单声道转换为处理的单声道信号"""
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        return np.ascontiguousarray(mono)


    def _update_input_buffers(self, mono: torch.Tensor):
        """轮换输入缓存并执行降采样到16kHz（mono 为 GPU tensor）"""
        # 输入wav缓冲区轮换：shift旧数据，写入新数据
        self.input_wav_work[:-self.block_samples].copy_(self.input_wav[self.block_samples:])
        self.input_wav_work[-self.block_samples:].zero_()
        self.input_wav, self.input_wav_work = self.input_wav_work, self.input_wav
        self.input_wav[-mono.shape[0]:] = mono

        # 降采样输入到16kHz供HuBERT特征提取使用
        self.input_wav_res_work[:-self.block_samples_16k].copy_(self.input_wav_res[self.block_samples_16k:])
        self.input_wav_res_work[-self.block_samples_16k:].zero_()
        self.input_wav_res, self.input_wav_res_work = self.input_wav_res_work, self.input_wav_res
        # 取额外 2*hz_centis 上下文喂 resampler（抵消重采样延迟），输出跳过前 160 样本
        # （resampler 预热延迟），写入 16k 缓存尾部
        resampler_in = self.input_wav[-mono.shape[0] - 2 * self.hz_centis:]
        resampler_out = self.resampler(resampler_in)[160:]
        target_len = 160 * (mono.shape[0] // self.hz_centis + 1)
        self.input_wav_res[-target_len:] = resampler_out

    def _run_inference(self):
        """执行语音转换推理或直通模式。参数从 InferenceConfig 传入，pipeline 无状态。"""
        if self.function == "vc" and self.runner:
            infer = self.runner.process_block(
                self.input_wav_res, self.runtime_params,
                self.block_samples_16k, self.skip_head, self.return_length,
            )
            if self.resampler_model2dev:
                infer = self.resampler_model2dev(infer)
        else:
            infer = self.input_wav[self.extra_samples:].clone()
        return infer

