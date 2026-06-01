"""RVC 实时语音转换 — PySide6 桌面GUI"""
import json
import os
import sys
import logging
import time
import threading
import queue
import traceback

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import librosa
from torchaudio.transforms import Resample as TatResample
import torchaudio
import torchaudio.functional as TAF

# 修复 Windows 终端中文乱码: 强制 UTF-8 输出
import io
if sys.stdout is not None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_log_handlers = [logging.StreamHandler(stream=sys.stdout)] if sys.stdout else []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QFileDialog, QMessageBox, QGroupBox, QLineEdit, QRadioButton,
    QButtonGroup, QTabWidget, QScrollArea, QFrame, QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPalette, QColor

import soundfile as sf
from configs.config import Config
from rvc.denoise import TorchGate

config = Config()

CONFIG_PATH = "configs/inuse/gui_config.json"
MODELS_PATH = "configs/models.json"


# ─────────────────── 模型列表数据 ───────────────────

class ModelListData:
    """管理模型列表的持久化"""

    @staticmethod
    def load():
        if not os.path.exists(MODELS_PATH): return []
        try:
            with open(MODELS_PATH, "r", encoding="utf-8") as f: return json.load(f).get("models", [])
        except: return []

    @staticmethod
    def save(models):
        os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
        with open(MODELS_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": models}, f, indent=2, ensure_ascii=False)


# ─────────────────── 模型卡片 ───────────────────

class ModelCard(QFrame):
    """模型卡片: 点击名称选中, 箭头展开参数"""

    load_requested = Signal(str, str, str, float, float, float, float)

    def __init__(self, name="", pth="", idx="", pitch=0, f0method="fcpe",
                 index_rate=0.0, rms_mix=0.0, gender=50, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._build(name, pth, idx, pitch, f0method, index_rate, rms_mix, gender)
        self._body.setVisible(False)

    def _build(self, name, pth, idx, pitch, f0method, index_rate, rms_mix, gender):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # 头部: 单选圆圈 + 名称 + 展开箭头
        hdr = QWidget(); hdr.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(4,3,4,3)
        self._radio = QRadioButton()
        self._radio.setAutoExclusive(False)
        self._radio.clicked.connect(self._on_load)
        self._name = QLabel(name or os.path.splitext(os.path.basename(pth))[0])
        self._name.setStyleSheet("font-weight:bold")
        self._arrow = QLabel("▶"); self._arrow.setStyleSheet("color:#666;font-size:9px")
        hl.addWidget(self._radio); hl.addWidget(self._name, 1)
        hl.addWidget(self._arrow)
        root.addWidget(hdr)
        self._name.mousePressEvent = lambda e: self._on_load()
        self._arrow.mousePressEvent = lambda e: self._toggle()

        # 内容 (展开后显示)
        self._body = QWidget()
        bl = QGridLayout(self._body); bl.setContentsMargins(24,2,6,4); bl.setSpacing(2)
        r = 0
        bl.addWidget(QLabel("模型"), r, 0)
        self.pth_edit = QLineEdit(pth); bl.addWidget(self.pth_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(22,20)
        b.clicked.connect(lambda: self._browse(self.pth_edit, "模型 (*.pth)")); bl.addWidget(b, r, 2); r+=1
        bl.addWidget(QLabel("索引"), r, 0)
        self.idx_edit = QLineEdit(idx); bl.addWidget(self.idx_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(22,20)
        b.clicked.connect(lambda: self._browse(self.idx_edit, "索引 (*.index)")); bl.addWidget(b, r, 2); r+=1

        def add_s(label, sl, lbl, row):
            bl.addWidget(QLabel(label), row, 0); bl.addWidget(sl, row, 1); bl.addWidget(lbl, row, 2)

        self.pit_sl = self._sl(-16,16,1,pitch); self.pit_lbl = QLabel(str(pitch))
        self.pit_sl.valueChanged.connect(lambda v: self.pit_lbl.setText(str(v)))
        add_s("音调", self.pit_sl, self.pit_lbl, r); r+=1
        self.gen_sl = self._sl(0,100,1,gender); self.gen_lbl = QLabel(f"{(gender/100-0.5)*4:+.2f}")
        self.gen_sl.valueChanged.connect(lambda v: self.gen_lbl.setText(f"{(v/100-0.5)*4:+.2f}"))
        add_s("性别", self.gen_sl, self.gen_lbl, r); r+=1
        self.ir_sl = self._sl(0,100,1,int(index_rate*100)); self.ir_lbl = QLabel(f"{index_rate:.2f}")
        self.ir_sl.valueChanged.connect(lambda v: self.ir_lbl.setText(f"{v/100:.2f}"))
        add_s("索引", self.ir_sl, self.ir_lbl, r); r+=1
        self.rms_sl = self._sl(0,100,1,int(rms_mix*100)); self.rms_lbl = QLabel(f"{rms_mix:.2f}")
        self.rms_sl.valueChanged.connect(lambda v: self.rms_lbl.setText(f"{v/100:.2f}"))
        add_s("响度", self.rms_sl, self.rms_lbl, r); r+=1

        self._del = QPushButton("删除此模型")
        self._del.setStyleSheet("QPushButton{background:#c0392b;color:white;border:none;padding:3px;border-radius:2px;font-size:11px}QPushButton:hover{background:#e74c3c}")
        bl.addWidget(self._del, r, 0, 1, 3)
        root.addWidget(self._body)
        self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")

    def _sl(self, mn, mx, st, dv):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv); return s

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path: tgt.setText(path)

    def _on_load(self):
        self._radio.setChecked(True)
        self.load_requested.emit(
            self._name.text(), self.pth_edit.text().strip(), self.idx_edit.text().strip(),
            self.pit_sl.value(), self.ir_sl.value()/100, self.rms_sl.value()/100,
            self.gen_sl.value()/100,
        )

    def get_data(self):
        return {
            "name": self._name.text(), "pth": self.pth_edit.text().strip(),
            "idx": self.idx_edit.text().strip(), "pitch": self.pit_sl.value(),
            "index_rate": self.ir_sl.value()/100,
            "rms_mix": self.rms_sl.value()/100, "gender": self.gen_sl.value()/100,
        }

    def set_active(self, active):
        self._radio.blockSignals(True); self._radio.setChecked(active); self._radio.blockSignals(False)
        if active:
            self.setStyleSheet("ModelCard{border:1px solid #28a745;border-radius:3px;margin:1px;background:rgba(40,167,69,0.06)}")
            self._name.setStyleSheet("font-weight:bold;color:#28a745")
        else:
            self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")
            self._name.setStyleSheet("font-weight:bold")


# ─────────────────── 预设 ───────────────────

PRESETS = {
    "原声纯净": {"eq_low": 0, "eq_mid": 0, "eq_high": 0, "warmth": 0, "compress": 0, "reverb": 0},
    "温暖电台": {"eq_low": 3, "eq_mid": 1.5, "eq_high": -1, "warmth": 0.35, "compress": 0.5, "reverb": 0.02},
    "贴耳ASMR": {"eq_low": 1, "eq_mid": -1, "eq_high": 4, "warmth": 0.1, "compress": 0.3, "reverb": 0.04},
    "明亮通透": {"eq_low": -2, "eq_mid": 2, "eq_high": 3.5, "warmth": 0.2, "compress": 0.25, "reverb": 0.1},
    "空旷大厅": {"eq_low": -1, "eq_mid": 0, "eq_high": 1.5, "warmth": 0, "compress": 0.15, "reverb": 0.35},
}


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1, device=a.device) + deltaphase
    t = torch.arange(n, device=a.device).unsqueeze(-1) / n
    return a * (fade_out**2) + b * (fade_in**2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n


def get_audio_devices(hostapi_name=None):
    sd._terminate(); sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for ha in hostapis:
        for idx in ha["devices"]:
            devices[idx]["hostapi_name"] = ha["name"]
    ha_names = [h["name"] for h in hostapis]
    if hostapi_name not in ha_names:
        hostapi_name = ha_names[0] if ha_names else ""
    filt = lambda d, ch: d[ch] > 0 and d.get("hostapi_name") == hostapi_name
    inputs = [d["name"] for d in devices if filt(d, "max_input_channels")]
    outputs = [d["name"] for d in devices if filt(d, "max_output_channels")]
    in_idx = [d["index"] for d in devices if filt(d, "max_input_channels")]
    out_idx = [d["index"] for d in devices if filt(d, "max_output_channels")]
    return ha_names, inputs, outputs, in_idx, out_idx


# ─────────────────── 实时引擎 ───────────────────

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
        dev_info = sd.query_devices(dev_idx)
        logger.info("副输出: %s (idx=%d, sr=%d)", dev_info["name"], dev_idx, int(dev_info["default_samplerate"]))
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
                except: pass
                try: s.close()
                except: pass
        self.stream = self.stream2 = None

    @staticmethod
    def _fast_rms(wav, fl, hop):
        p = fl // 2
        sq = F.pad(wav.unsqueeze(0).unsqueeze(0), (p, p), mode='reflect') ** 2
        return torch.sqrt(torch.clamp(F.avg_pool1d(sq, fl, hop).squeeze(), min=1e-8))

    def _cb(self, indata, outdata, frames, times, status):
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
                try:
                    if self.out2_q.full():
                        try: self.out2_q.get_nowait()
                        except: pass
                    self.out2_q.put_nowait(outdata.copy())
                except Exception as e: logger.warning("副输出队列写入失败: %s", e)

        self.infer_ms = (time.perf_counter() - t0) * 1000


# ─────────────────── 运行时参数 ───────────────────

class Params:
    threshold = -60; pitch = 0; index_rate = 0.0; rms_mix = 0.0; gender = 0.0
    f0method = "fcpe"; I_nr = False; O_nr = False; use_pv = False
    enable_eq = False; eq_low = 0; eq_mid = 0; eq_high = 0
    warmth = 0; compress = 0; reverb = 0
    bgm_enable = False; bgm_vol = 0.5
    enable_out2 = False

p = Params()
engine = RealtimeEngine()


class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, pth, idx, idx_rate): super().__init__(); self.pth=pth; self.idx=idx; self.rate=idx_rate
    def run(self):
        try: self.ok.emit(engine.load_model(self.pth, self.idx, self.rate, True))
        except Exception as e: self.err.emit(str(e))


# ─────────────────── 离线推理 ───────────────────

_FFMPEG = os.path.join(os.getcwd(), "ffmpeg.exe")
_X_PAD = 3  # 与 Config 中 x_pad 一致


class OfflineWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_path, output_path, pth, idx, idx_rate, pitch, f0method, rms_mix):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.pth = pth; self.idx = idx; self.idx_rate = idx_rate
        self.pitch = pitch; self.f0method = f0method; self.rms_mix = rms_mix

    def run(self):
        try:
            self._do_run()
        except Exception:
            self.error.emit(traceback.format_exc().strip().splitlines()[-1])

    def _do_run(self):
        # 加载音频
        self.progress.emit(0, 100)
        wav, sr = self._load_audio(self.input_path)

        # 重采样到 16kHz（HuBERT 要求）
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        self.progress.emit(10, 100)

        # 推理引擎
        from rvc.realtime_engine import RealtimeVC
        vc = RealtimeVC(config, self.pth, self.idx, self.idx_rate)
        vc.load()
        vc.change_key(self.pitch)
        self.progress.emit(20, 100)

        tgt_sr = vc.tgt_sr
        t_pad = 16000 * _X_PAD  # 48000 样本 (16kHz 下 3s)
        t_pad_tgt = tgt_sr * _X_PAD  # 模型采样率下的 padding

        # Pad 音频（反射填充）
        audio_pad = np.pad(wav, (t_pad, t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // 160  # 帧数

        # F0 提取
        pitch_coarse, pitchf = None, None
        if vc.if_f0 == 1:
            pitch_coarse, pitchf = vc._get_f0(
                torch.from_numpy(audio_pad).float(), self.pitch, self.f0method
            )
            pitch_coarse = pitch_coarse[:p_len].unsqueeze(0)
            pitchf = pitchf[:p_len].unsqueeze(0)
        self.progress.emit(40, 100)

        # HuBERT 特征提取
        feats = torch.from_numpy(audio_pad).float()
        if vc.is_half:
            feats = feats.half()
        feats = feats.view(1, -1).to(config.device)
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=config.device)
        with torch.no_grad():
            logits = vc.model.extract_features(
                source=feats, padding_mask=padding_mask, output_layer=12
            )
            feats = logits[0]
        feats = torch.cat((feats, feats[:, -1:, :]), 1)
        self.progress.emit(55, 100)

        # FAISS 索引匹配
        if vc.index is not None and vc.index_rate > 0:
            try:
                npy = feats[0].cpu().numpy().astype("float32")
                score, ix = vc.index.search(npy, k=min(8, vc.index.ntotal))
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(vc.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                    if vc.is_half:
                        npy = npy.astype("float16")
                    feats = (
                        torch.from_numpy(npy).unsqueeze(0).to(config.device) * vc.index_rate
                        + (1 - vc.index_rate) * feats
                    )
            except Exception:
                logger.debug("索引匹配失败: %s", traceback.format_exc())
        self.progress.emit(65, 100)

        # 上采样特征
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]

        # 合成器推理（不传 skip_head / return_length）
        p_len_t = torch.LongTensor([p_len]).to(config.device)
        sid = torch.LongTensor([0]).to(config.device)
        with torch.no_grad():
            if vc.if_f0 == 1:
                result = vc.net_g.infer(feats, p_len_t, pitch_coarse, pitchf, sid)
            else:
                result = vc.net_g.infer(feats, p_len_t, sid)
        audio1 = result[0][0, 0].data.cpu().float().numpy()
        self.progress.emit(85, 100)

        # Trim padding
        audio1 = audio1[t_pad_tgt : -t_pad_tgt] if t_pad_tgt > 0 else audio1

        # RMS 响度匹配
        if self.rms_mix != 1:
            audio1 = self._change_rms(wav, 16000, audio1, tgt_sr, self.rms_mix)

        # 归一化 + 保存
        audio_max = np.abs(audio1).max() / 0.99
        if audio_max > 1:
            audio1 = audio1 / audio_max
        sf.write(self.output_path, audio1, tgt_sr, subtype="FLOAT")
        self.progress.emit(100, 100)
        self.finished.emit(self.output_path)

    @staticmethod
    def _change_rms(data1, sr1, data2, sr2, rate):
        """参照 pipeline.py 的 change_rms — data1 是输入, data2 是输出, rate 是输出占比"""
        rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
        rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
        rms1 = torch.from_numpy(rms1)
        rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
        rms2 = torch.from_numpy(rms2)
        rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
        rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
        data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
        return data2

    def _load_audio(self, path):
        """加载任意格式音频 → (mono_float32, sample_rate)"""
        import warnings
        path = os.path.abspath(path)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*PySoundFile.*")
                warnings.filterwarnings("ignore", message=".*audioread.*", category=FutureWarning)
                return librosa.load(path, sr=None, mono=True)
        except Exception:
            pass
        if not os.path.exists(_FFMPEG):
            raise FileNotFoundError(f"找不到 ffmpeg: {_FFMPEG}\n也无法用 librosa 加载: {path}")
        import subprocess, re
        info = subprocess.run([_FFMPEG, "-i", path], capture_output=True, text=True)
        sr = 48000
        for line in info.stderr.split('\n'):
            if 'Hz' in line and 'Audio' in line:
                m = re.search(r'(\d+) Hz', line)
                if m: sr = int(m.group(1)); break
        cmd = [_FFMPEG, "-i", path, "-vn", "-acodec", "pcm_f32le", "-f", "wav", "-ac", "1", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode:
            raise RuntimeError("ffmpeg 解码失败")
        raw = np.frombuffer(proc.stdout, dtype=np.float32)
        return raw.astype(np.float32), sr


# ─────────────────── 主窗口 ───────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self._active_card = None
        self._loading = False
        self._off_worker = None
        self._timer = QTimer(); self._timer.timeout.connect(lambda: self.stat_lbl.setText(f"推理: {int(engine.infer_ms)}"))
        self._build_ui()
        self._load_cfg()

    def _sl(self, mn, mx, st, dv):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv); return s

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(4); root.setContentsMargins(6,6,6,6)

        tabs = QTabWidget()
        tabs.addTab(self._build_settings_tab(), "设置")
        tabs.addTab(self._build_models_tab(), "模型")
        tabs.addTab(self._build_audio_tab(), "声学")
        tabs.addTab(self._build_offline_tab(), "离线")
        root.addWidget(tabs)

        # 底部控制栏
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self.btn_start = QPushButton("开始")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_start); ctrl.addWidget(self.btn_stop)
        self.model_lbl = QLabel("当前: -"); self.model_lbl.setMinimumWidth(140)
        self.delay_lbl = QLabel("延迟: -"); self.delay_lbl.setMinimumWidth(70)
        self.stat_lbl = QLabel("推理: -"); self.stat_lbl.setMinimumWidth(80)
        ctrl.addWidget(self.model_lbl); ctrl.addStretch()
        ctrl.addWidget(self.delay_lbl); ctrl.addWidget(self.stat_lbl)
        root.addLayout(ctrl)

    def _build_settings_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1)
        r = 0

        # 设备
        g.addWidget(QLabel("音频驱动"), r, 0)
        self.ha_combo = QComboBox(); self.ha_combo.currentTextChanged.connect(self._ha_changed)
        g.addWidget(self.ha_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("麦克风"), r, 0)
        self.in_combo = QComboBox(); g.addWidget(self.in_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("主输出"), r, 0)
        self.out_combo = QComboBox(); g.addWidget(self.out_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("副输出"), r, 0)
        self.out2_combo = QComboBox(); g.addWidget(self.out2_combo, r, 1)
        br = QPushButton("刷新"); br.setFixedWidth(45); br.clicked.connect(self._reload_dev)
        g.addWidget(br, r, 2); r+=1
        self.sr_r1 = QRadioButton(); self.sr_r1.setChecked(True)
        self.sr_r2 = QRadioButton()
        self.sr_r1_lbl = QLabel("模型采样率: -")
        self.sr_r2_lbl = QLabel("设备采样率: -")
        sr = QHBoxLayout(); sr.setSpacing(4)
        sr.addWidget(self.sr_r1); sr.addWidget(self.sr_r1_lbl)
        sr.addSpacing(12); sr.addWidget(self.sr_r2); sr.addWidget(self.sr_r2_lbl); sr.addStretch()
        g.addLayout(sr, r, 0, 1, 3); r+=1

        # 分隔
        g.addWidget(self._sep(), r, 0, 1, 3); r+=1

        # 引擎参数
        def add_sl(label, sl, lbl, row):
            g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

        self.th_sl = self._sl(-60,0,1,-60); self.th_lbl = QLabel("-60"); self.th_lbl.setMinimumWidth(35)
        self.th_sl.valueChanged.connect(lambda v: (self.th_lbl.setText(str(v)), setattr(p,'threshold',v)))
        add_sl("响应阈值", self.th_sl, self.th_lbl, r); r+=1
        self.bl_sl = self._sl(2,150,1,25); self.bl_lbl = QLabel("0.25"); self.bl_lbl.setMinimumWidth(35)
        self.bl_sl.valueChanged.connect(lambda v: self.bl_lbl.setText(f"{v/100:.2f}"))
        add_sl("采样长度", self.bl_sl, self.bl_lbl, r); r+=1
        self.cf_sl = self._sl(1,15,1,5); self.cf_lbl = QLabel("0.05"); self.cf_lbl.setMinimumWidth(35)
        self.cf_sl.valueChanged.connect(lambda v: self.cf_lbl.setText(f"{v/100:.2f}"))
        add_sl("淡入淡出", self.cf_sl, self.cf_lbl, r); r+=1
        self.ex_sl = self._sl(5,500,1,250); self.ex_lbl = QLabel("2.50"); self.ex_lbl.setMinimumWidth(35)
        self.ex_sl.valueChanged.connect(lambda v: self.ex_lbl.setText(f"{v/100:.2f}"))
        add_sl("额外上下文", self.ex_sl, self.ex_lbl, r); r+=1

        g.addWidget(QLabel("音高算法"), r, 0)
        self.f0_combo = QComboBox(); self.f0_combo.addItems(["fcpe", "rmvpe"]); self.f0_combo.setFixedWidth(80)
        g.addWidget(self.f0_combo, r, 1)
        return w

    def _build_models_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(4); l.setContentsMargins(10,10,10,10)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("模型列表")); bar.addStretch()
        btn_add = QPushButton("+ 添加模型")
        btn_add.setStyleSheet("QPushButton{background:#007acc;color:white;padding:3px 10px;border-radius:2px;font-weight:bold}QPushButton:hover{background:#005f9e}")
        btn_add.clicked.connect(self._add_model)
        bar.addWidget(btn_add)
        l.addLayout(bar)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        self._models_layout = QVBoxLayout(container); self._models_layout.addStretch()
        scroll.setWidget(container)
        l.addWidget(scroll, 1)
        self._model_cards = []
        return w

    def _build_audio_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1)
        r = 0

        # 降噪
        self.inr = QCheckBox("输入降噪"); self.ounr = QCheckBox("输出降噪")
        nr = QHBoxLayout(); nr.addWidget(self.inr); nr.addWidget(self.ounr); nr.addStretch()
        g.addLayout(nr, r, 0, 1, 3); r+=1
        g.addWidget(self._sep(), r, 0, 1, 3); r+=1

        # 音效
        self.eq_en = QCheckBox("开启音效")
        self.preset_combo = QComboBox(); self.preset_combo.addItems(PRESETS.keys()); self.preset_combo.setFixedWidth(90)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        row0 = QHBoxLayout(); row0.addWidget(self.eq_en); row0.addWidget(self.preset_combo); row0.addStretch()
        g.addLayout(row0, r, 0, 1, 3); r+=1

        def add_eq(label, sl, lbl, row):
            g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

        self.eq_lo = self._sl(-3000,2000,500,0); self.eq_lo_lbl = QLabel("0.0"); self.eq_lo_lbl.setMinimumWidth(35)
        self.eq_lo.valueChanged.connect(lambda v: self.eq_lo_lbl.setText(f"{v/100:.1f}"))
        add_eq("低频增益", self.eq_lo, self.eq_lo_lbl, r); r+=1
        self.eq_mi = self._sl(-2000,2000,500,0); self.eq_mi_lbl = QLabel("0.0"); self.eq_mi_lbl.setMinimumWidth(35)
        self.eq_mi.valueChanged.connect(lambda v: self.eq_mi_lbl.setText(f"{v/100:.1f}"))
        add_eq("中频增益", self.eq_mi, self.eq_mi_lbl, r); r+=1
        self.eq_hi = self._sl(-3000,3000,500,0); self.eq_hi_lbl = QLabel("0.0"); self.eq_hi_lbl.setMinimumWidth(35)
        self.eq_hi.valueChanged.connect(lambda v: self.eq_hi_lbl.setText(f"{v/100:.1f}"))
        add_eq("高频增益", self.eq_hi, self.eq_hi_lbl, r); r+=1
        self.warm_sl = self._sl(0,100,1,0); self.warm_lbl = QLabel("0.00"); self.warm_lbl.setMinimumWidth(35)
        self.warm_sl.valueChanged.connect(lambda v: self.warm_lbl.setText(f"{v/100:.2f}"))
        add_eq("电子管饱和", self.warm_sl, self.warm_lbl, r); r+=1
        self.comp_sl = self._sl(0,100,5,0); self.comp_lbl = QLabel("0.00"); self.comp_lbl.setMinimumWidth(35)
        self.comp_sl.valueChanged.connect(lambda v: self.comp_lbl.setText(f"{v/100:.2f}"))
        add_eq("动态压限", self.comp_sl, self.comp_lbl, r); r+=1
        self.rev_sl = self._sl(0,50,1,0); self.rev_lbl = QLabel("0.00"); self.rev_lbl.setMinimumWidth(35)
        self.rev_sl.valueChanged.connect(lambda v: self.rev_lbl.setText(f"{v/100:.2f}"))
        add_eq("空间混响", self.rev_sl, self.rev_lbl, r)
        return w

    def _build_offline_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1); r = 0

        g.addWidget(QLabel("输入文件"), r, 0)
        self.off_in = QLineEdit(); g.addWidget(self.off_in, r, 1)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: self._off_browse(self.off_in, "in"))
        g.addWidget(b, r, 2); r += 1

        g.addWidget(QLabel("输出文件"), r, 0)
        self.off_out = QLineEdit(); g.addWidget(self.off_out, r, 1)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: self._off_browse(self.off_out, "out"))
        g.addWidget(b, r, 2); r += 1

        row = QHBoxLayout()
        self.off_btn = QPushButton("开始转换")
        self.off_btn.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:5px 16px;border-radius:3px}QPushButton:hover{background:#005f9e}QPushButton:disabled{background:#555}")
        self.off_btn.clicked.connect(self._off_start)
        row.addWidget(self.off_btn)
        self.off_status = QLabel("")
        row.addWidget(self.off_status)
        row.addStretch()
        g.addLayout(row, r, 0, 1, 3); r += 1

        self.off_progress = QProgressBar(); self.off_progress.setValue(0)
        g.addWidget(self.off_progress, r, 0, 1, 3)
        return w

    @staticmethod
    def _sep():
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet("color:#444"); return f

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "assets/weights", "模型 (*.pth)")
        if not path: return
        name = os.path.splitext(os.path.basename(path))[0]
        self._add_card(name=name, pth=path)

    def _add_card(self, name="", pth="", idx="", pitch=0, f0method="fcpe",
                  index_rate=0.0, rms_mix=0.0, gender=50):
        card = ModelCard(name, pth, idx, pitch, f0method, index_rate, rms_mix, gender)
        card.load_requested.connect(self._on_card_load)
        card._del.clicked.connect(lambda: self._remove_card(card))
        self._models_layout.insertWidget(self._models_layout.count()-1, card)
        self._model_cards.append(card)
        return card

    def _remove_card(self, card):
        if self._active_card == card: self._active_card = None
        self._model_cards.remove(card)
        self._models_layout.removeWidget(card)
        card.deleteLater()
        self._save_models()

    def _on_card_load(self, name, pth, idx, pitch, ir, rms, gender):
        if not pth: return
        # 只选中, 不加载. 点击"开始"时才加载
        if self._active_card: self._active_card.set_active(False)
        for c in self._model_cards:
            if c.pth_edit.text().strip() == pth:
                self._active_card = c; c.set_active(True); break
        self.model_lbl.setText(f"当前: {name}")

    def _save_models(self):
        models = [c.get_data() for c in self._model_cards]
        ModelListData.save(models)

    def _apply_preset(self, name):
        if name not in PRESETS: return
        pr = PRESETS[name]
        self.eq_lo.setValue(int(pr["eq_low"]*100)); self.eq_mi.setValue(int(pr["eq_mid"]*100)); self.eq_hi.setValue(int(pr["eq_high"]*100))
        self.warm_sl.setValue(int(pr["warmth"]*100)); self.comp_sl.setValue(int(pr["compress"]*100)); self.rev_sl.setValue(int(pr["reverb"]*100))

    # ── 配置持久化 ──

    def _load_cfg(self):
        # 加载模型列表
        for m in ModelListData.load():
            self._add_card(m.get("name",""), m.get("pth",""), m.get("idx",""),
                           m.get("pitch",0), m.get("f0method","fcpe"),
                           m.get("index_rate",0), m.get("rms_mix",0),
                           int(m.get("gender",0.5)*100))
        # 加载全局配置
        d = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f: d = json.load(f)
            except: pass
        # 恢复设备路由（需要先填充设备列表）
        self._reload_dev()
        ha = d.get("ha", "")
        if ha:
            idx = self.ha_combo.findText(ha)
            if idx >= 0: self.ha_combo.setCurrentIndex(idx)
        for dev_key, combo in [("in_dev", self.in_combo), ("out_dev", self.out_combo)]:
            dev = d.get(dev_key, "")
            if dev:
                idx = combo.findText(dev)
                if idx >= 0: combo.setCurrentIndex(idx)
        # 副输出 (索引0="不启用", 所以+1偏移)
        out2_dev = d.get("out2_dev", "")
        if out2_dev:
            idx = self.out2_combo.findText(out2_dev)
            if idx >= 0: self.out2_combo.setCurrentIndex(idx)
        if d.get("sr_mode", "model") == "device": self.sr_r2.setChecked(True)
        # F0 算法
        f0 = d.get("f0", "fcpe")
        idx = self.f0_combo.findText(f0)
        if idx >= 0: self.f0_combo.setCurrentIndex(idx)
        self.th_sl.setValue(d.get("th", -60))
        self.bl_sl.setValue(int(d.get("bl", 0.25)*100))
        self.cf_sl.setValue(int(d.get("cf", 0.05)*100))
        self.ex_sl.setValue(int(d.get("ex", 2.5)*100))
        self.inr.setChecked(d.get("inr", False)); self.ounr.setChecked(d.get("ounr", False))
        self.eq_en.setChecked(d.get("eq_en", False))
        self.eq_lo.setValue(int(d.get("eq_lo", 0)*100)); self.eq_mi.setValue(int(d.get("eq_mi", 0)*100))
        self.eq_hi.setValue(int(d.get("eq_hi", 0)*100)); self.warm_sl.setValue(int(d.get("warm", 0)*100))
        self.comp_sl.setValue(int(d.get("comp", 0)*100)); self.rev_sl.setValue(int(d.get("rev", 0)*100))
        pr = d.get("preset", "原声纯净")
        if pr in PRESETS: self.preset_combo.setCurrentText(pr)
        # 恢复选中的模型
        selected = d.get("selected", "")
        if selected:
            for c in self._model_cards:
                if c.pth_edit.text().strip() == selected:
                    self._active_card = c; c.set_active(True)
                    self.model_lbl.setText(f"当前: {c._name.text()}")
                    break

    def _save_cfg(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        self._save_models()
        d = {
            "version": 2, "th": self.th_sl.value(), "bl": self.bl_sl.value()/100,
            "cf": self.cf_sl.value()/100, "ex": self.ex_sl.value()/100,
            "inr": self.inr.isChecked(), "ounr": self.ounr.isChecked(),
            "f0": self.f0_combo.currentText(),
            "eq_en": self.eq_en.isChecked(),
            "eq_lo": self.eq_lo.value()/100, "eq_mi": self.eq_mi.value()/100,
            "eq_hi": self.eq_hi.value()/100, "warm": self.warm_sl.value()/100,
            "comp": self.comp_sl.value()/100, "rev": self.rev_sl.value()/100,
            "preset": self.preset_combo.currentText(),
            "ha": self.ha_combo.currentText(),
            "in_dev": self.in_combo.currentText(),
            "out_dev": self.out_combo.currentText(),
            "out2_dev": self.out2_combo.currentText(),
            "sr_mode": "model" if self.sr_r1.isChecked() else "device",
            "selected": self._active_card.pth_edit.text().strip() if self._active_card else "",
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, ensure_ascii=False)

    # ── 设备管理 ──

    def _reload_dev(self):
        self.ha_combo.blockSignals(True); self.ha_combo.clear()
        names, *_ = get_audio_devices(); self.ha_combo.addItems(names)
        self.ha_combo.blockSignals(False); self._ha_changed(self.ha_combo.currentText())

    def _ha_changed(self, name):
        if not name: return
        _, ins, outs, _, _ = get_audio_devices(name)
        self.in_combo.clear(); self.in_combo.addItems(ins)
        self.out_combo.clear(); self.out_combo.addItems(outs)
        self.out2_combo.clear()
        self.out2_combo.addItem("不启用")
        self.out2_combo.addItems(outs)

    # ── 启动/停止 ──

    def _start(self):
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在模型列表中选择一个模型")
            return
        pth = self._active_card.pth_edit.text().strip()
        if not pth: QMessageBox.warning(self, "提示", "模型文件路径为空"); return
        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value()/100
        p.pitch = self._active_card.pit_sl.value()
        p.index_rate = ir; p.rms_mix = self._active_card.rms_sl.value()/100
        p.gender = (self._active_card.gen_sl.value()/100 - 0.5) * 4  # 0~100 → -2~+2 半音
        p.f0method = self.f0_combo.currentText()
        name = self._active_card._name.text()
        # logger.info("选择: %s (pitch=%d, f0=%s, ir=%.2f)", name, p.pitch, p.f0method, ir)
        self._start_engine(pth, idx, ir)

    def _start_engine(self, pth, idx, idx_rate):
        if self._loading: return
        self._loading = True
        self.btn_start.setEnabled(False); self.btn_start.setText("加载中...")
        self._lt = LoadThread(pth, idx, idx_rate)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        if hasattr(self, '_lt') and self._lt:
            self._lt.deleteLater(); self._lt = None

    def _on_loaded(self, sr):
        try:
            _, _, _, in_idx, out_idx = get_audio_devices(self.ha_combo.currentText())
            p.threshold = self.th_sl.value()
            p.I_nr = self.inr.isChecked(); p.O_nr = self.ounr.isChecked(); p.use_pv = False
            p.enable_eq = self.eq_en.isChecked(); p.eq_low = self.eq_lo.value()/100
            p.eq_mid = self.eq_mi.value()/100; p.eq_hi = self.eq_hi.value()/100
            p.warmth = self.warm_sl.value()/100; p.compress = self.comp_sl.value()/100
            p.reverb = self.rev_sl.value()/100
            p.bgm_enable = False; p.enable_out2 = self.out2_combo.currentIndex() > 0
            sr_type = "sr_model" if self.sr_r1.isChecked() else "sr_device"
            engine.setup(sr_type, in_idx[self.in_combo.currentIndex()], out_idx[self.out_combo.currentIndex()],
                         self.bl_sl.value()/100, self.cf_sl.value()/100, self.ex_sl.value()/100, p)
            engine.bgm_audio = None; engine.bgm_ptr = 0
            if p.enable_out2:
                engine.setup_out2(out_idx[self.out2_combo.currentIndex() - 1])
            self.sr_r1_lbl.setText(f"模型采样率: {engine.sr_model}")
            self.sr_r2_lbl.setText(f"设备采样率: {engine.sr_dev}")
            dl = (engine.stream.latency[-1] if engine.stream else 0) + self.bl_sl.value()/100 + self.cf_sl.value()/100 + 0.01
            if p.I_nr: dl += min(self.cf_sl.value()/100, 0.04)
            self.delay_lbl.setText(f"延迟: {int(dl*1000)}")
            self.btn_start.setEnabled(False); self.btn_start.setText("运行中")
            self.btn_stop.setEnabled(True)
            self._timer.start(200); self._save_cfg()
            # logger.info("启动完成 (设备:%d 模型:%d)", engine.sr_dev, engine.sr_model)
        except Exception as e: self._on_err(str(e))

    def _on_err(self, e):
        self.btn_start.setEnabled(True); self.btn_start.setText("开始")
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "错误", str(e))

    def _stop(self):
        if not engine.running: return
        self._timer.stop(); engine.stop()
        self.btn_start.setEnabled(True); self.btn_start.setText("开始")
        self.btn_stop.setEnabled(False)
        self.delay_lbl.setText("延迟: -"); self.stat_lbl.setText("推理: -")
        logger.info("停止")

    # ── 离线推理 ──

    def _off_browse(self, tgt, kind):
        if kind == "in":
            path, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频 (*.wav *.mp3 *.flac *.ogg *.m4a *.wma *.aac *.opus);;所有 (*)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存音频", "", "WAV (*.wav)")
        if path:
            tgt.setText(path)
            if kind == "in" and not self.off_out.text():
                base, _ = os.path.splitext(path)
                self.off_out.setText(base + "_converted.wav")

    def _off_start(self):
        inp = self.off_in.text().strip()
        out = self.off_out.text().strip()
        if not inp:
            QMessageBox.warning(self, "提示", "请选择输入文件"); return
        if not os.path.exists(inp):
            QMessageBox.warning(self, "提示", f"文件不存在: {inp}"); return
        if not out:
            base, _ = os.path.splitext(inp); out = base + "_converted.wav"
            self.off_out.setText(out)
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在「模型」中选择一个模型"); return
        pth = self._active_card.pth_edit.text().strip()
        if not pth:
            QMessageBox.warning(self, "提示", "模型路径为空"); return
        if engine.running:
            QMessageBox.warning(self, "提示", "请先停止实时变声"); return

        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value() / 100
        pitch = self._active_card.pit_sl.value()
        f0m = self.f0_combo.currentText()
        rms = self._active_card.rms_sl.value() / 100

        self._off_worker = OfflineWorker(inp, out, pth, idx, ir, pitch, f0m, rms)
        self._off_worker.progress.connect(self._off_progress)
        self._off_worker.finished.connect(self._off_done)
        self._off_worker.error.connect(self._off_err)
        self.off_btn.setEnabled(False); self.off_btn.setText("转换中...")
        self.off_progress.setValue(0)
        self._off_worker.start()

    def _off_progress(self, cur, total):
        self.off_progress.setMaximum(total)
        self.off_progress.setValue(cur)
        self.off_status.setText(f"{cur}/{total}")

    def _off_done(self, path):
        self.off_btn.setEnabled(True); self.off_btn.setText("开始转换")
        self.off_status.setText(f"完成: {path}")
        if self._off_worker:
            self._off_worker.wait(); self._off_worker = None

    def _off_err(self, msg):
        self.off_btn.setEnabled(True); self.off_btn.setText("开始转换")
        self.off_status.setText("错误")
        if self._off_worker:
            self._off_worker.wait(); self._off_worker = None
        QMessageBox.critical(self, "离线推理错误", msg)

    def closeEvent(self, e):
        if self._off_worker and self._off_worker.isRunning():
            self._off_worker.quit(); self._off_worker.wait()
        self._save_cfg(); engine.stop(); e.accept()


def set_dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(40,40,40)); p.setColor(QPalette.WindowText, QColor(220,220,220))
    p.setColor(QPalette.Base, QColor(30,30,30)); p.setColor(QPalette.Text, QColor(220,220,220))
    p.setColor(QPalette.Button, QColor(55,55,55)); p.setColor(QPalette.ButtonText, QColor(220,220,220))
    p.setColor(QPalette.Highlight, QColor(66,133,244)); p.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(p)
    app.setStyleSheet("QGroupBox{font-weight:bold;margin-top:6px}QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px}QSlider{min-height:18px}QSlider::groove:horizontal{height:4px}QSlider::handle:horizontal{width:12px;margin:-5px 0}")

if __name__ == "__main__":
    app = QApplication(sys.argv); set_dark(app)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
