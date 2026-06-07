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
from torchaudio.transforms import Resample as TatResample
import torchaudio.functional as TAF

from rvc.params import p
from rvc.audio_utils import phase_vocoder
from rvc.denoise import TorchGate
from configs.config import Config

logger = logging.getLogger(__name__)
config = Config()


class RealtimeEngine:
    def __init__(self):
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

        self.input_wav = None; self.input_wav_res = None; self.input_wav_denoise = None
        self.sola_buffer = None; self.nr_buffer = None; self.output_buffer = None
        self.rms_buffer = None; self.fade_in = None; self.fade_out = None
        self.resampler = None; self.resampler2 = None; self.tg = None
        self.reverb_buffer = None
        self.bgm_audio = None; self.bgm_ptr = 0

        self.loaded_pth = ""; self.loaded_idx = ""
        self.infer_ms = 0.0

    def load_model(self, pth, idx, idx_rate, force=False):
        if not force and self.vc_engine and self.loaded_pth == pth and self.loaded_idx == idx:
            self.vc_engine.change_index_rate(idx_rate)
            return self.vc_engine.tgt_sr
        from rvc.realtime_engine import RealtimeVC
        self.vc_engine = RealtimeVC(config, pth, idx, idx_rate)
        self.vc_engine.load()
        self.loaded_pth = pth; self.loaded_idx = idx
        return self.vc_engine.tgt_sr

    def setup(self, sr_type, in_dev, out_dev, block_t, cf_t, extra_t, p):
        if self.running:
            self.stop()
        sd.default.device = [in_dev, out_dev]
        self.sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        self.sr_model = self.vc_engine.tgt_sr
        self.sr = self.sr_dev

        in_info, out_info = sd.query_devices(in_dev), sd.query_devices(out_dev)
        self.channels = min(int(in_info["max_input_channels"]), int(out_info["max_output_channels"]), 2)

        zc = self.sr_dev // 100
        self.block_frame = int(np.round(block_t * self.sr_dev / zc)) * zc
        self.crossfade_frame = int(np.round(cf_t * self.sr_dev / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = int(np.round(extra_t * self.sr_dev / zc)) * zc

        self.block_frame_16k = 160 * self.block_frame // zc
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc

        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(n, device=config.device)
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * n // zc, device=config.device)
        self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        self.zc = zc

        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=config.device)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.reverb_buffer = torch.zeros(int(0.15 * self.sr_dev), device=config.device)

        ls = torch.linspace(0, 1, steps=self.sola_buffer_frame, device=config.device)
        self.fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self.fade_out = 1 - self.fade_in

        self.resampler = TatResample(self.sr_dev, 16000, dtype=torch.float32).to(config.device)
        if self.sr_model != self.sr_dev:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr_dev, dtype=torch.float32).to(config.device)
        else:
            self.resampler_model2dev = None

        self.tg = TorchGate(sr=self.sr_dev, n_fft=4 * zc, prop_decrease=0.9).to(config.device)

        self.stream = sd.Stream(callback=self._cb, blocksize=self.block_frame, samplerate=self.sr_dev, channels=self.channels, dtype="float32")
        self.stream.start()
        self.running = True

    def setup_out2(self, dev_idx):
        self.stream2 = sd.OutputStream(device=dev_idx, samplerate=self.sr, channels=self.channels, dtype="float32", latency='low')
        self.stream2.start()
        while not self.out2_q.empty(): self.out2_q.get()
        threading.Thread(target=self._out2_worker, daemon=True).start()

    def _out2_worker(self):
        while self.running:
            try:
                d = self.out2_q.get(timeout=0.01)
                if self.stream2 and self.stream2.active: self.stream2.write(d)
            except queue.Empty: pass
            except Exception as e: logger.warning("副输出写入失败: %s", e)

    def stop(self):
        self.running = False
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
        except Exception as e:
            logger.error("音频回调异常: %s", e, exc_info=True)
            outdata[:] = 0

    def _cb_impl(self, indata, outdata, frames, times, status):
        t0 = time.perf_counter()
        with torch.no_grad():
            mono = librosa.to_mono(indata.T) if indata.ndim > 1 else indata[:, 0]
            if p.threshold > -60:
                tmp = np.append(self.rms_buffer, mono)
                rms = librosa.feature.rms(y=tmp, frame_length=4*self.zc, hop_length=self.zc)[:, 2:]
                self.rms_buffer[:] = tmp[-4*self.zc:]
                tmp = tmp[2*self.zc - self.zc//2:]
                db = librosa.amplitude_to_db(rms, ref=1.0)[0] < p.threshold
                for i in range(db.shape[0]):
                    if db[i]: tmp[i*self.zc:(i+1)*self.zc] = 0
                mono = tmp[self.zc//2:]

            self.input_wav = torch.roll(self.input_wav, -self.block_frame)
            self.input_wav[-mono.shape[0]:] = torch.from_numpy(mono.copy()).to(config.device)
            self.input_wav_res = torch.roll(self.input_wav_res, -self.block_frame_16k)

            if p.I_nr:
                self.input_wav_denoise = torch.roll(self.input_wav_denoise, -self.block_frame)
                iw = self.input_wav[-self.sola_buffer_frame - self.block_frame:]
                iw = self.tg(iw.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
                iw[:self.sola_buffer_frame] *= self.fade_in
                iw[:self.sola_buffer_frame] += self.nr_buffer * self.fade_out
                self.input_wav_denoise[-self.block_frame:] = iw[:self.block_frame]
                self.nr_buffer[:] = iw[self.block_frame:]
                self.input_wav_res[-self.block_frame_16k-160:] = self.resampler(self.input_wav_denoise[-self.block_frame-2*self.zc:])[160:]
            else:
                self.input_wav_res[-160*(mono.shape[0]//self.zc+1):] = self.resampler(self.input_wav[-mono.shape[0]-2*self.zc:])[160:]

            if self.function == "vc" and self.vc_engine:
                self.vc_engine.change_key(p.pitch)
                self.vc_engine.change_index_rate(p.index_rate)
                self.vc_engine.change_formant(p.gender)
                infer = self.vc_engine.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, p.f0method)
                if self.resampler_model2dev:
                    infer = self.resampler_model2dev(infer)
            elif p.I_nr:
                infer = self.input_wav_denoise[self.extra_frame:].clone()
            else:
                infer = self.input_wav[self.extra_frame:].clone()

            if p.O_nr and self.function == "vc":
                self.output_buffer = torch.roll(self.output_buffer, -self.block_frame)
                self.output_buffer[-self.block_frame:] = infer[-self.block_frame:]
                infer = self.tg(infer.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

            if p.rms_mix < 1 and self.function == "vc":
                ref = self.input_wav_denoise[self.extra_frame:] if p.I_nr else self.input_wav[self.extra_frame:]
                r1 = self._fast_rms(ref[:infer.shape[0]], 4*self.zc, self.zc)
                r1 = F.interpolate(r1[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = self._fast_rms(infer, 4*self.zc, self.zc)
                r2 = F.interpolate(r2[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = torch.max(r2, torch.ones_like(r2)*1e-3)
                infer *= torch.pow(r1 / r2, 1 - p.rms_mix)

            if p.enable_eq:
                sr = self.sr
                if p.eq_low: infer = TAF.bass_biquad(infer.unsqueeze(0), sr, gain=p.eq_low).squeeze(0)
                if p.eq_mid: infer = TAF.equalizer_biquad(infer.unsqueeze(0), sr, 1000.0, 0.707, p.eq_mid).squeeze(0)
                if p.eq_high: infer = TAF.treble_biquad(infer.unsqueeze(0), sr, gain=p.eq_high).squeeze(0)
                if p.warmth > 0:
                    d = 1 + p.warmth * 3; m = 1 + p.warmth * 0.5
                    infer = torch.tanh(infer * d) / d * m

            ci = infer[None, None, :self.sola_buffer_frame + self.sola_search_frame]
            cn = F.conv1d(ci, self.sola_buffer[None, None, :])
            cd = torch.sqrt(F.conv1d(ci**2, torch.ones(1,1,self.sola_buffer_frame, device=config.device)) + 1e-8)
            off = torch.argmax(cn[0,0] / cd[0,0])
            infer = infer[off:]
            if not p.use_pv:
                infer[:self.sola_buffer_frame] *= self.fade_in
                infer[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out
            else:
                infer[:self.sola_buffer_frame] = phase_vocoder(self.sola_buffer, infer[:self.sola_buffer_frame], self.fade_out, self.fade_in)
            self.sola_buffer[:] = infer[self.block_frame:self.block_frame+self.sola_buffer_frame]
            chunk = infer[:self.block_frame]

            if p.enable_eq:
                if p.reverb > 0:
                    ri = torch.cat([self.reverb_buffer, chunk])
                    d1 = int(0.017*self.sr); d2 = int(0.031*self.sr); d3 = int(0.047*self.sr); d4 = int(0.073*self.sr)
                    t1 = ri[-self.block_frame-d1:-d1]*0.3
                    t2 = -ri[-self.block_frame-d2:-d2]*0.2
                    t3 = ri[-self.block_frame-d3:-d3]*0.15
                    t4 = -ri[-self.block_frame-d4:-d4]*0.08
                    rv = t1+t2+t3+t4
                    rv = F.avg_pool1d(rv[None,None], 5, 1, 2).squeeze()
                    chunk = chunk*(1-p.reverb*0.5) + rv*p.reverb
                    self.reverb_buffer = ri[-self.reverb_buffer.shape[0]:]
                if p.compress > 0:
                    th = 1 - p.compress * 0.6
                    ab = torch.abs(chunk)
                    chunk = torch.where(ab > th, torch.sign(chunk)*(th+(1-th)*torch.tanh((ab-th)/(1-th))), chunk)

            if p.bgm_enable and self.bgm_audio is not None and p.bgm_vol > 0:
                bl = self.bgm_audio.shape[0]; need = self.block_frame; ci2 = 0
                bc = torch.zeros(self.block_frame)
                while need > 0:
                    take = min(need, bl - self.bgm_ptr)
                    bc[ci2:ci2+take] = self.bgm_audio[self.bgm_ptr:self.bgm_ptr+take]
                    self.bgm_ptr = (self.bgm_ptr + take) % bl; ci2 += take; need -= take
                chunk += bc.to(config.device) * p.bgm_vol

            outdata[:] = chunk.repeat(self.channels, 1).t().cpu().numpy()
            if self.stream2 and p.enable_out2:
                if self.out2_q.full():
                    try: self.out2_q.get_nowait()
                    except Exception: pass
                self.out2_q.put_nowait(outdata.copy())

        self.infer_ms = (time.perf_counter() - t0) * 1000
