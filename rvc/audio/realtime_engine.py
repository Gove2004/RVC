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

from rvc.audio.denoise import SpectralSubtraction
from rvc.audio.output_router import route_secondary_output, write_main_output
from rvc.audio.realtime_mix import apply_rms_mix
from rvc.audio.sola import apply_sola
from rvc.inference.params import HUBERT_DEFAULT
from rvc.runtime import Config

logger = logging.getLogger(__name__)
config = Config()

# 实时推理错误容忍配置
MAX_CONSECUTIVE_ERRORS = 3  # 连续错误达到此阈值后停止推理


class RealtimeEngine:
    def __init__(self, runtime_params, inference_cache=None, on_runtime_error=None):
        self.runtime_params = runtime_params
        self.inference_cache = inference_cache
        self.on_runtime_error = on_runtime_error
        self.pipeline = None
        self.stream = None
        self.stream2 = None
        self.running = False
        self.function = "vc"
        self.out2_q = queue.Queue(maxsize=10)

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
        self.nr_ss = None            # 效果器（setup 时创建）
        self._last_nr_params = None  # 降噪参数缓存（用于检测变化）

        self.pth_path = ""
        self.infer_ms = 0.0
        self.measure_ms = 0.0  # 硬件时间戳实测端到端延迟（EMA 平滑）
        self.error_count = 0
        self.max_error_count = MAX_CONSECUTIVE_ERRORS
        self.last_error = ""
        self.runtime_error_pending = False

    def load_model(self, pth, force=False, hubert=HUBERT_DEFAULT):
        if not force and self.pipeline and self.pth_path == pth:
            return self.pipeline.target_sr
        from rvc.inference.pipeline import VCPipeline
        try:
            self.pipeline = VCPipeline(config, pth, self.inference_cache, hubert=hubert)
            self.pipeline.load()
            self.pth_path = pth
            return self.pipeline.target_sr
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.pipeline = None
            raise

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t):
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
        self.channels = min(int(in_info["max_input_channels"]), int(out_info["max_output_channels"]), 2)

        self._init_processing(self.sr, block_t, cf_t, extra_t, self.channels)

        # 开流前预热推理：首次推理会触发 CUDA Graph 捕获（每模型 3 次 warmup 前向 + capture，
        # 单块可能数百 ms），提前用静音数据跑完，让首次真实回调即为热状态。
        # 这就是「停止后重新开始延迟变低」的原因——图已捕获；现在把它提前到开流前。
        self.warmup_inference(2)

        try:
            self.stream = sd.Stream(callback=self._cb, blocksize=self.block_samples, samplerate=self.sr, channels=self.channels, dtype="float32")
            self.stream.start()
            self.running = True
        except Exception as e:
            if "Invalid sample rate" in str(e) or "-9997" in str(e):
                raise RuntimeError(f"采样率 {self.sr} Hz 不支持，请切换到「模型采样率」或 MME 驱动") from e
            raise

    def _init_processing(self, sr, block_t, cf_t, extra_t, channels):
        """设备无关的处理状态初始化（采样率/块大小/缓存/重采样/降噪）。

        setup() 与离线文件流式推理 process_file() 共用，保证两条路径算法完全一致。
        需先设置 self.sr_model（= pipeline.target_sr），用于判断是否需模型→设备重采样。
        """
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
        self.input_wav = torch.zeros(n, device=config.device)
        self.input_wav_res = torch.zeros(160 * n // zc, device=config.device)
        self.input_wav_work = torch.empty_like(self.input_wav)
        self.input_wav_res_work = torch.empty_like(self.input_wav_res)
        self.hz_centis = zc
        # 输入侧 pinned buffer（CPU↔GPU 非阻塞拷贝复用，避免每块分配）
        self._in_pin = torch.empty(self.block_samples, dtype=torch.float32, pin_memory=True)

        self.sola_buffer = torch.zeros(self.sola_buffer_samples, device=config.device)

        ls = torch.linspace(0, 1, steps=self.sola_buffer_samples, device=config.device)
        self.fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self.fade_out = 1 - self.fade_in
        self.sola_norm_kernel = torch.ones(1, 1, self.sola_buffer_samples, device=config.device)

        self.resampler = TatResample(self.sr, 16000, dtype=torch.float32).to(config.device)
        if self.sr_model != self.sr:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr, dtype=torch.float32).to(config.device)
        else:
            self.resampler_model2dev = None

        # 降噪器（输入侧）
        self.nr_ss = SpectralSubtraction(self.sr)
        self.nr_ss.reset()
        self._last_nr_params = None

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
        try:
            with torch.no_grad():
                for _ in range(n):
                    self._cb_impl(indata, outdata, frames, None, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning("推理预热失败（不影响运行）: %s", e)

    def setup_out2(self, dev_idx):
        dev_name = ""
        try:
            dev_name = sd.query_devices(dev_idx)["name"]
        except Exception:
            pass
        logger.info(f"设置副输出设备: {dev_idx}（{dev_name}）" if dev_name else f"设置副输出设备: {dev_idx}")
        def out2_callback(outdata, frames, time_info, status):
            try:
                if not self.out2_q.empty():
                    data = self.out2_q.get_nowait()
                    outdata[:] = data[:frames]
                else:
                    outdata[:] = 0
            except Exception:
                outdata[:] = 0
        try:
            self.stream2 = sd.OutputStream(
                device=dev_idx, samplerate=self.sr, channels=self.channels,
                dtype="float32", blocksize=self.block_samples, callback=out2_callback
            )
            self.stream2.start()
            logger.info(f"副输出流已启动: 采样率={self.sr}, 声道={self.channels}, blocksize={self.block_samples}")
            while not self.out2_q.empty():
                try:
                    self.out2_q.get_nowait()
                except queue.Empty:
                    pass
        except Exception as e:
            if "Invalid sample rate" in str(e) or "-9997" in str(e):
                raise RuntimeError(f"副输出采样率 {self.sr} Hz 不支持") from e
            raise


    def stop(self):
        self.running = False
        self.error_count = 0
        self.runtime_error_pending = False
        for s in (self.stream2, self.stream):
            if s:
                try:
                    s.abort()
                except Exception as e:
                    logger.debug("停止流时出错: %s", e)
                try:
                    s.close()
                except Exception as e:
                    logger.debug("关闭流时出错: %s", e)
        self.stream = self.stream2 = None

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
        from rvc.inference.params import Params

        self.sr_model = self.pipeline.target_sr
        tgt_sr = self.sr_model
        wav = self._load_audio_at_sr(input_path, tgt_sr)

        if params is None:
            params = Params()
        self.runtime_params = params
        if f0method is not None:
            self.runtime_params.f0method = f0method
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
        import librosa
        import numpy as np

        from rvc.audio.loader import load_audio_native
        wav, src_sr = load_audio_native(input_path)
        if src_sr != tgt_sr:
            wav = librosa.resample(wav, orig_sr=src_sr, target_sr=tgt_sr)
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
        p_enable_out2 = params.enable_out2
        p_nr_enable = params.nr_enable
        p_nr_strength = params.nr_strength

        with torch.no_grad():
            # ── 阶段1-2: 输入准备 + 降噪 + 缓存轮换 ─────────────────
            mono = self._prepare_input(indata)
            # 输入一次性上 GPU（降噪/推理/输出共用，避免 CPU↔GPU 往返）。
            # 预分配 pinned buffer + 非阻塞拷贝，避免每块重复分配临时张量。
            src = torch.from_numpy(mono)
            self._in_pin[: src.shape[0]].copy_(src, non_blocking=True)
            mono = self._in_pin[: src.shape[0]].to(config.device, non_blocking=True)
            mono = self._apply_denoise(mono, p_nr_enable, p_nr_strength)
            self._update_input_buffers(mono)

            # ── 阶段3: 语音转换推理 ───────────────────────────────────
            infer = self._run_inference()

            # ── 阶段4: RMS音量包络混合 ────────────────────────────────
            if p_rms_mix < 1 and self.function == "vc":
                infer = self._apply_rms_mix(infer, p_rms_mix)

            # ── 阶段5: SOLA对齐 ───────────────────────────────────────
            chunk = self._apply_sola(infer)

            # ── 阶段6: 输出写入与副输出路由 ───────────────────────────
            self._write_output(chunk, outdata, p_enable_out2)

        self.infer_ms = (time.perf_counter() - t0) * 1000

    def _apply_sola(self, infer):
        """SOLA 时间拉伸对齐，输出块长度 = block_samples"""
        return apply_sola(
            infer, self.sola_buffer, self.sola_norm_kernel,
            self.fade_in, self.fade_out,
            self.block_samples, self.sola_buffer_samples,
            self.sola_search_samples,
        )

    def _write_output(self, chunk, outdata, enable_out2):
        """主输出写入 + 副输出路由"""
        write_main_output(chunk, outdata, self.channels)
        route_secondary_output(outdata, self.stream2, self.out2_q, enable_out2)

    def _prepare_input(self, indata):
        """将输入的立体声/单声道转换为处理的单声道信号"""
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        return np.ascontiguousarray(mono)

    def _apply_denoise(self, mono: torch.Tensor, enable: bool, strength: float) -> torch.Tensor:
        """输入侧降噪（谱减法），输入输出均为 GPU tensor，避免 CPU↔GPU 往返。

        参数变化检测：强度变了才更新效果器。
        """
        if enable and self.nr_ss is not None:
            if self._last_nr_params != strength:
                self.nr_ss.set_strength(strength)
                self._last_nr_params = strength
            return self.nr_ss(mono)
        return mono

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
        self.input_wav_res[-160*(mono.shape[0]//self.hz_centis+1):] = self.resampler(self.input_wav[-mono.shape[0]-2*self.hz_centis:])[160:]

    def _run_inference(self):
        """执行语音转换推理或直通模式"""
        if self.function == "vc" and self.pipeline:
            self.pipeline.configure(
                pitch=self.runtime_params.pitch,
                gender=self.runtime_params.gender,
                break_enable=self.runtime_params.break_enable,
                break_src_hz=self.runtime_params.break_src_hz,
            )
            infer = self.pipeline.infer(
                self.input_wav_res, self.block_samples_16k,
                self.skip_head, self.return_length,
                self.runtime_params.f0method, self.runtime_params.protect
            )
            if self.resampler_model2dev:
                infer = self.resampler_model2dev(infer)
        else:
            infer = self.input_wav[self.extra_samples:].clone()
        return infer

    def _apply_rms_mix(self, infer, rms_mix):
        """应用RMS音量包络混合，使转换音量参考原始音量"""
        ref = self.input_wav[self.extra_samples:]
        return apply_rms_mix(ref, infer, rms_mix, self.hz_centis)
