"""实时音频引擎 — 管理 sounddevice 流、缓冲区、SOLA、声学效果"""
import logging
import queue
import threading
import time

import librosa
import numpy as np
import sounddevice as sd
import torch
from torchaudio.transforms import Resample as TatResample

from rvc.audio.effects import create_realtime_chain
from rvc.audio.output_router import mix_bgm, route_secondary_output, write_main_output
from rvc.audio.realtime_effects import apply_post_sola_effects, apply_pre_sola_effects
from rvc.audio.realtime_mix import apply_rms_mix
from rvc.audio.sola import apply_sola
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
        self.vc_engine = None
        self.stream = None
        self.stream2 = None
        self.running = False
        self.function = "vc"
        self.out2_q = queue.Queue(maxsize=10)

        self.sr = 48000; self.zc = 480; self.channels = 1
        self.block_frame = 0; self.block_frame_16k = 0
        self.crossfade_frame = 0; self.sola_buffer_frame = 0
        self.sola_search_frame = 0; self.extra_frame = 0
        self.skip_head = 0; self.return_length = 0

        self.input_wav = None; self.input_wav_res = None
        self.input_wav_work = None; self.input_wav_res_work = None
        self.sola_buffer = None; self.output_buffer = None
        self.fade_in = None; self.fade_out = None; self.sola_norm_kernel = None
        self.bgm_mix_buffer = None
        self.resampler = None; self.resampler2 = None
        self.bgm_audio = None; self.bgm_ptr = 0

        # 效果器（setup 时创建）
        self.eq = None
        self.reverb = None

        # 效果参数缓存（用于检测变化）
        self._last_eq_params = None
        self._last_reverb_mix = None

        self.loaded_pth = ""; self.loaded_idx = ""
        self.infer_ms = 0.0
        self.error_count = 0
        self.max_error_count = MAX_CONSECUTIVE_ERRORS
        self.last_error = ""
        self.runtime_error_pending = False

    def load_model(self, pth, idx, idx_rate, force=False):
        if not force and self.vc_engine and self.loaded_pth == pth and self.loaded_idx == idx:
            self.vc_engine.change_index_rate(idx_rate)
            return self.vc_engine.tgt_sr
        from rvc.inference.pipeline import VCPipeline
        try:
            self.vc_engine = VCPipeline(config, pth, idx, idx_rate, self.inference_cache)
            self.vc_engine.load()
            self.loaded_pth = pth; self.loaded_idx = idx
            return self.vc_engine.tgt_sr
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.vc_engine = None
            raise

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t):
        if self.running:
            self.stop()
        self.error_count = 0
        self.last_error = ""
        self.runtime_error_pending = False
        sd.default.device = [in_dev, out_dev]
        self.sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        self.sr_model = self.vc_engine.tgt_sr
        self.sr = self.sr_model if sr_type == "sr_model" else self.sr_dev

        in_info, out_info = sd.query_devices(in_dev), sd.query_devices(out_dev)
        self.channels = min(int(in_info["max_input_channels"]), int(out_info["max_output_channels"]), 2)

        zc = self.sr // 100
        self.block_frame = int(np.round(block_t * self.sr / zc)) * zc
        self.crossfade_frame = int(np.round(cf_t * self.sr / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = int(np.round(extra_t * self.sr / zc)) * zc

        self.block_frame_16k = 160 * self.block_frame // zc
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc

        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(n, device=config.device)
        self.input_wav_res = torch.zeros(160 * n // zc, device=config.device)
        self.input_wav_work = torch.empty_like(self.input_wav)
        self.input_wav_res_work = torch.empty_like(self.input_wav_res)
        self.bgm_mix_buffer = torch.empty(self.block_frame, device=config.device)
        self.zc = zc

        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=config.device)
        self.output_buffer = self.input_wav.clone()

        ls = torch.linspace(0, 1, steps=self.sola_buffer_frame, device=config.device)
        self.fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self.fade_out = 1 - self.fade_in
        self.sola_norm_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=config.device)

        self.resampler = TatResample(self.sr, 16000, dtype=torch.float32).to(config.device)
        if self.sr_model != self.sr:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr, dtype=torch.float32).to(config.device)
        else:
            self.resampler_model2dev = None

        # 创建效果器（实时模式）
        _, self.eq, self.reverb = create_realtime_chain(self.sr)

        try:
            self.stream = sd.Stream(callback=self._cb, blocksize=self.block_frame, samplerate=self.sr, channels=self.channels, dtype="float32")
            self.stream.start()
            self.running = True
        except Exception as e:
            if "Invalid sample rate" in str(e) or "-9997" in str(e):
                raise RuntimeError(f"采样率 {self.sr} Hz 不支持，请切换到「模型采样率」或 MME 驱动") from e
            raise

    def setup_out2(self, dev_idx):
        logger.info(f"设置副输出设备: {dev_idx}")
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
                dtype="float32", blocksize=self.block_frame, callback=out2_callback
            )
            self.stream2.start()
            logger.info(f"副输出流已启动: 采样率={self.sr}, 声道={self.channels}, blocksize={self.block_frame}")
            while not self.out2_q.empty():
                try: self.out2_q.get_nowait()
                except: pass
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

    def _cb(self, indata, outdata, frames, times, status):
        try:
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

    def _cb_impl(self, indata, outdata, frames, times, status):
        t0 = time.perf_counter()
        params = self.runtime_params
        with torch.no_grad():
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
            mono = np.ascontiguousarray(mono)

            self.input_wav_work[:-self.block_frame].copy_(self.input_wav[self.block_frame:])
            self.input_wav_work[-self.block_frame:].zero_()
            self.input_wav, self.input_wav_work = self.input_wav_work, self.input_wav
            self.input_wav[-mono.shape[0]:] = torch.from_numpy(mono).to(config.device)
            self.input_wav_res_work[:-self.block_frame_16k].copy_(self.input_wav_res[self.block_frame_16k:])
            self.input_wav_res_work[-self.block_frame_16k:].zero_()
            self.input_wav_res, self.input_wav_res_work = self.input_wav_res_work, self.input_wav_res
            self.input_wav_res[-160*(mono.shape[0]//self.zc+1):] = self.resampler(self.input_wav[-mono.shape[0]-2*self.zc:])[160:]

            if self.function == "vc" and self.vc_engine:
                self.vc_engine.change_key(params.pitch)
                self.vc_engine.change_index_rate(params.index_rate)
                self.vc_engine.change_formant(params.gender)
                infer = self.vc_engine.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, params.f0method, params.protect)
                if self.resampler_model2dev:
                    infer = self.resampler_model2dev(infer)
            else:
                infer = self.input_wav[self.extra_frame:].clone()

            if params.rms_mix < 1 and self.function == "vc":
                ref = self.input_wav[self.extra_frame:]
                infer = apply_rms_mix(ref, infer, params.rms_mix, self.zc)

            infer, self._last_eq_params = apply_pre_sola_effects(
                infer,
                params,
                self.eq,
                self._last_eq_params,
            )

            chunk = apply_sola(
                infer,
                self.sola_buffer,
                self.sola_norm_kernel,
                self.fade_in,
                self.fade_out,
                self.block_frame,
                self.sola_buffer_frame,
                self.sola_search_frame,
                params.use_pv,
            )

            chunk, self._last_reverb_mix = apply_post_sola_effects(
                chunk,
                params,
                self.reverb,
                self._last_reverb_mix,
            )

            if params.bgm_enable:
                chunk, self.bgm_audio, self.bgm_ptr = mix_bgm(
                    chunk,
                    self.bgm_audio,
                    self.bgm_ptr,
                    self.bgm_mix_buffer,
                    params.bgm_vol,
                    self.block_frame,
                )

            write_main_output(chunk, outdata, self.channels)
            should_log_out2_disabled = route_secondary_output(outdata, self.stream2, self.out2_q, params.enable_out2)
            if should_log_out2_disabled and not hasattr(self, '_out2_debug_logged'):
                logger.warning("副输出流存在但 enable_out2=False")
                self._out2_debug_logged = True

        self.infer_ms = (time.perf_counter() - t0) * 1000
