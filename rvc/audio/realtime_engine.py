"""实时音频引擎 — 管理 sounddevice 流、缓冲区、SOLA、声学效果"""
import logging
import queue
import threading
import time

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from PySide6.QtCore import QObject, Signal
from torchaudio.transforms import Resample as TatResample

from rvc.audio.utils import phase_vocoder
from rvc.audio.effects import create_realtime_chain
from gui.configs import Config

logger = logging.getLogger(__name__)
config = Config()


class EngineSignals(QObject):
    runtime_error = Signal(str)


class RealtimeEngine:
    def __init__(self, runtime_params, inference_cache=None):
        self.runtime_params = runtime_params
        self.inference_cache = inference_cache
        self.vc_engine = None
        self.stream = None
        self.stream2 = None
        self.running = False
        self.function = "vc"
        self.out2_q = queue.Queue(maxsize=10)
        self.signals = EngineSignals()

        self.sr = 48000; self.zc = 480; self.channels = 1
        self.block_frame = 0; self.block_frame_16k = 0
        self.crossfade_frame = 0; self.sola_buffer_frame = 0
        self.sola_search_frame = 0; self.extra_frame = 0
        self.skip_head = 0; self.return_length = 0

        self.input_wav = None; self.input_wav_res = None
        self.sola_buffer = None; self.output_buffer = None
        self.fade_in = None; self.fade_out = None
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
        self.max_error_count = 3
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
        self.zc = zc

        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=config.device)
        self.output_buffer = self.input_wav.clone()

        ls = torch.linspace(0, 1, steps=self.sola_buffer_frame, device=config.device)
        self.fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self.fade_out = 1 - self.fade_in

        self.resampler = TatResample(self.sr, 16000, dtype=torch.float32).to(config.device)
        if self.sr_model != self.sr:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr, dtype=torch.float32).to(config.device)
        else:
            self.resampler_model2dev = None

        # 创建效果器（实时模式）
        _, self.eq, self.reverb = create_realtime_chain(self.sr)

        self.stream = sd.Stream(callback=self._cb, blocksize=self.block_frame, samplerate=self.sr, channels=self.channels, dtype="float32")
        self.stream.start()
        self.running = True

    def setup_out2(self, dev_idx):
        def out2_callback(outdata, frames, time_info, status):
            try:
                if not self.out2_q.empty():
                    data = self.out2_q.get_nowait()
                    outdata[:] = data[:frames]
                else:
                    outdata[:] = 0
            except Exception:
                outdata[:] = 0
        self.stream2 = sd.OutputStream(
            device=dev_idx, samplerate=self.sr, channels=self.channels,
            dtype="float32", blocksize=self.block_frame, callback=out2_callback
        )
        self.stream2.start()
        while not self.out2_q.empty():
            try: self.out2_q.get_nowait()
            except: pass


    def stop(self):
        self.running = False
        self.error_count = 0
        self.runtime_error_pending = False
        for s in (self.stream2, self.stream):
            if s:
                try: s.abort()
                except Exception: pass
                try: s.close()
                except Exception: pass
        self.stream = self.stream2 = None

    @staticmethod
    def _fast_rms(wav, fl, hop):
        p = fl // 2
        sq = F.pad(wav.unsqueeze(0).unsqueeze(0), (p, p), mode='reflect') ** 2
        return torch.sqrt(torch.clamp(F.avg_pool1d(sq, fl, hop).squeeze(), min=1e-8))

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
                try:
                    self.signals.runtime_error.emit(self.last_error or "实时推理失败")
                except Exception:
                    logger.exception("发送实时错误信号失败")

    def _cb_impl(self, indata, outdata, frames, times, status):
        t0 = time.perf_counter()
        params = self.runtime_params
        with torch.no_grad():
            mono = librosa.to_mono(indata.T) if indata.ndim > 1 else indata[:, 0]

            self.input_wav = torch.roll(self.input_wav, -self.block_frame)
            self.input_wav[-mono.shape[0]:] = torch.from_numpy(mono.copy()).to(config.device)
            self.input_wav_res = torch.roll(self.input_wav_res, -self.block_frame_16k)
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
                r1 = self._fast_rms(ref[:infer.shape[0]], 4*self.zc, self.zc)
                r1 = F.interpolate(r1[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = self._fast_rms(infer, 4*self.zc, self.zc)
                r2 = F.interpolate(r2[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = torch.max(r2, torch.ones_like(r2)*1e-3)
                infer *= torch.pow(r1 / r2, 1 - params.rms_mix)

            # 应用 EQ（SOLA 之前，只在参数变化时同步）
            if params.enable_eq and self.eq:
                current_eq = (params.eq_sub, params.eq_low, params.eq_mid, params.eq_hi_mid, params.eq_high)
                if self._last_eq_params != current_eq:
                    self.eq.set_band('sub', params.eq_sub)
                    self.eq.set_band('low', params.eq_low)
                    self.eq.set_band('mid', params.eq_mid)
                    self.eq.set_band('hi_mid', params.eq_hi_mid)
                    self.eq.set_band('high', params.eq_high)
                    self._last_eq_params = current_eq
                infer = self.eq(infer)

            ci = infer[None, None, :self.sola_buffer_frame + self.sola_search_frame]
            cn = F.conv1d(ci, self.sola_buffer[None, None, :])
            cd = torch.sqrt(F.conv1d(ci**2, torch.ones(1,1,self.sola_buffer_frame, device=config.device)) + 1e-8)
            off = torch.argmax(cn[0,0] / cd[0,0])
            infer = infer[off:]
            if not params.use_pv:
                infer[:self.sola_buffer_frame] *= self.fade_in
                infer[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out
            else:
                infer[:self.sola_buffer_frame] = phase_vocoder(self.sola_buffer, infer[:self.sola_buffer_frame], self.fade_out, self.fade_in)
            self.sola_buffer[:] = infer[self.block_frame:self.block_frame+self.sola_buffer_frame]
            chunk = infer[:self.block_frame]

            # 应用混响（SOLA 之后，只在参数变化时同步）
            if params.enable_eq and self.reverb and params.reverb > 0:
                if self._last_reverb_mix != params.reverb:
                    self.reverb.set_mix(params.reverb)
                    self._last_reverb_mix = params.reverb
                chunk = self.reverb(chunk)

            if params.bgm_enable and self.bgm_audio is not None and params.bgm_vol > 0:
                bl = self.bgm_audio.shape[0]; need = self.block_frame; ci2 = 0
                bc = torch.zeros(self.block_frame)
                while need > 0:
                    take = min(need, bl - self.bgm_ptr)
                    bc[ci2:ci2+take] = self.bgm_audio[self.bgm_ptr:self.bgm_ptr+take]
                    self.bgm_ptr = (self.bgm_ptr + take) % bl; ci2 += take; need -= take
                chunk += bc.to(config.device) * params.bgm_vol

            outdata[:] = chunk.repeat(self.channels, 1).t().cpu().numpy()
            if self.stream2 and params.enable_out2:
                if self.out2_q.full():
                    try: self.out2_q.get_nowait()
                    except Exception: pass
                self.out2_q.put_nowait(outdata.copy())

        self.infer_ms = (time.perf_counter() - t0) * 1000
