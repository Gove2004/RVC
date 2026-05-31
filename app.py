"""RVC 实时语音转换 — PySide6 桌面GUI"""
import json
import os
import sys
import logging
import time
import threading
import queue

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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QFileDialog, QMessageBox, QGroupBox, QLineEdit, QRadioButton,
    QButtonGroup, QTabWidget,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPalette, QColor

from configs.config import Config
from rvc.denoise import TorchGate

config = Config()

CONFIG_PATH = "configs/inuse/gui_config.json"

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

    def setup(self, sr_type, in_dev, out_dev, wasapi, block_t, cf_t, extra_t, p):
        sd.default.device = [in_dev, out_dev]
        # 设备采样率和模型采样率分离
        self.sr_dev = int(sd.query_devices(in_dev)["default_samplerate"])
        self.sr_model = self.vc_engine.tgt_sr  # 40000 or 48000
        self.sr = self.sr_dev  # 音频 I/O 用设备采样率

        in_info, out_info = sd.query_devices(in_dev), sd.query_devices(out_dev)
        self.channels = min(int(in_info["max_input_channels"]), int(out_info["max_output_channels"]), 2)

        # 帧数计算基于设备采样率
        zc = self.sr_dev // 100  # 每10ms的采样数
        self.block_frame = int(np.round(block_t * self.sr_dev / zc)) * zc
        self.crossfade_frame = int(np.round(cf_t * self.sr_dev / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = int(np.round(extra_t * self.sr_dev / zc)) * zc

        # 16kHz 帧数
        self.block_frame_16k = 160 * self.block_frame // zc
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc

        # 缓冲区 (设备采样率)
        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(n, device=config.device)
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * n // zc, device=config.device)
        self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        self.zc = zc

        # SOLA 缓冲区 (设备采样率，保证交叉淡化正确)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=config.device)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.reverb_buffer = torch.zeros(int(0.15 * self.sr_dev), device=config.device)

        ls = torch.linspace(0, 1, steps=self.sola_buffer_frame, device=config.device)
        self.fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self.fade_out = 1 - self.fade_in

        # 重采样器
        self.resampler = TatResample(self.sr_dev, 16000, dtype=torch.float32).to(config.device)
        if self.sr_model != self.sr_dev:
            self.resampler_model2dev = TatResample(self.sr_model, self.sr_dev, dtype=torch.float32).to(config.device)
        else:
            self.resampler_model2dev = None

        self.tg = TorchGate(sr=self.sr_dev, n_fft=4 * zc, prop_decrease=0.9).to(config.device)

        ext = sd.WasapiSettings(exclusive=True) if "WASAPI" in sd.query_hostapis()[sd.query_devices(in_dev)["hostapi"]]["name"] and wasapi else None
        self.stream = sd.Stream(callback=self._cb, blocksize=self.block_frame, samplerate=self.sr_dev, channels=self.channels, dtype="float32", extra_settings=ext)
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

    def stop(self):
        self.running = False
        for s in (self.stream, self.stream2):
            if s: s.abort(); s.close()
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
            # 阈值降噪
            if p.threshold > -60:
                tmp = np.append(self.rms_buffer, mono)
                rms = librosa.feature.rms(y=tmp, frame_length=4*self.zc, hop_length=self.zc)[:, 2:]
                self.rms_buffer[:] = tmp[-4*self.zc:]
                tmp = tmp[2*self.zc - self.zc//2:]
                db = librosa.amplitude_to_db(rms, ref=1.0)[0] < p.threshold
                for i in range(db.shape[0]):
                    if db[i]: tmp[i*self.zc:(i+1)*self.zc] = 0
                mono = tmp[self.zc//2:]

            # 缓冲
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

            # 推理 (输出在模型采样率 sr_model 上)
            if self.function == "vc" and self.vc_engine:
                self.vc_engine.change_key(p.pitch)
                self.vc_engine.f0_method = p.f0method
                self.vc_engine.change_index_rate(p.index_rate)
                infer = self.vc_engine.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, p.f0method)
                # 重采样: 模型采样率 → 设备采样率
                if self.resampler_model2dev:
                    infer = self.resampler_model2dev(infer)
            elif p.I_nr:
                infer = self.input_wav_denoise[self.extra_frame:].clone()
            else:
                infer = self.input_wav[self.extra_frame:].clone()

            # 输出降噪
            if p.O_nr and self.function == "vc":
                self.output_buffer = torch.roll(self.output_buffer, -self.block_frame)
                self.output_buffer[-self.block_frame:] = infer[-self.block_frame:]
                infer = self.tg(infer.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

            # 响度因子
            if p.rms_mix < 1 and self.function == "vc":
                ref = self.input_wav_denoise[self.extra_frame:] if p.I_nr else self.input_wav[self.extra_frame:]
                r1 = self._fast_rms(ref[:infer.shape[0]], 4*self.zc, self.zc)
                r1 = F.interpolate(r1[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = self._fast_rms(infer, 4*self.zc, self.zc)
                r2 = F.interpolate(r2[None,None], size=infer.shape[0]+1, mode='linear', align_corners=True)[0,0,:-1]
                r2 = torch.max(r2, torch.ones_like(r2)*1e-3)
                infer *= torch.pow(r1 / r2, 1 - p.rms_mix)

            # 前置EQ + 饱和
            if p.enable_eq:
                sr = self.sr
                if p.eq_low: infer = TAF.bass_biquad(infer.unsqueeze(0), sr, gain=p.eq_low).squeeze(0)
                if p.eq_mid: infer = TAF.equalizer_biquad(infer.unsqueeze(0), sr, 1000.0, 0.707, p.eq_mid).squeeze(0)
                if p.eq_high: infer = TAF.treble_biquad(infer.unsqueeze(0), sr, gain=p.eq_high).squeeze(0)
                if p.warmth > 0:
                    d = 1 + p.warmth * 3; m = 1 + p.warmth * 0.5
                    infer = torch.tanh(infer * d) / d * m

            # SOLA
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

            # 后置: 混响 + 压缩
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

            # BGM
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
                    except: pass
                self.out2_q.put_nowait(outdata.copy())

        self.infer_ms = (time.perf_counter() - t0) * 1000


# ─────────────────── 运行时参数 ───────────────────

class Params:
    threshold = -60; pitch = 0; formant = 0.0; index_rate = 0.0; rms_mix = 0.0
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


# ─────────────────── 主窗口 ───────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self._timer = QTimer(); self._timer.timeout.connect(lambda: self.stat_lbl.setText(f"推理: {int(engine.infer_ms)} ms"))
        self._build_ui()
        self._load_cfg()

    def _sl(self, mn, mx, st, dv):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv); return s

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QVBoxLayout(cw); ml.setSpacing(3); ml.setContentsMargins(4,4,4,4)

        tabs = QTabWidget()
        tabs.addTab(self._tab_audio(), "音频设置")
        tabs.addTab(self._tab_engine(), "变声引擎")
        tabs.addTab(self._tab_master(), "声学母带")
        ml.addWidget(tabs)

        # 控制面板
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始转换")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:4px 10px}")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:4px 10px}")
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_start); ctrl.addWidget(self.btn_stop)
        self.func_grp = QButtonGroup()
        rb_m = QRadioButton("监听原声"); rb_v = QRadioButton("输出变声"); rb_v.setChecked(True)
        self.func_grp.addButton(rb_m, 0); self.func_grp.addButton(rb_v, 1)
        self.func_grp.idToggled.connect(lambda i, c: setattr(engine, 'function', 'im' if i==0 and c else 'vc') if c else None)
        ctrl.addWidget(rb_m); ctrl.addWidget(rb_v)
        self.delay_lbl = QLabel("延迟: - ms"); self.stat_lbl = QLabel("推理: - ms")
        ctrl.addWidget(self.delay_lbl); ctrl.addWidget(self.stat_lbl)
        ml.addLayout(ctrl)

    def _tab_audio(self):
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(3); l.setContentsMargins(4,4,4,4)

        g1 = QGroupBox("模型与索引"); gl = QGridLayout(g1); gl.setSpacing(2); gl.setContentsMargins(4,8,4,4)
        gl.addWidget(QLabel("模型文件"), 0, 0)
        self.pth_edit = QLineEdit(); gl.addWidget(self.pth_edit, 0, 1)
        b = QPushButton("选择"); b.clicked.connect(lambda: self._browse(self.pth_edit, "*.pth")); gl.addWidget(b, 0, 2)
        gl.addWidget(QLabel("特征索引"), 1, 0)
        self.idx_edit = QLineEdit(); gl.addWidget(self.idx_edit, 1, 1)
        b = QPushButton("选择"); b.clicked.connect(lambda: self._browse(self.idx_edit, "*.index")); gl.addWidget(b, 1, 2)
        l.addWidget(g1)

        g2 = QGroupBox("设备路由"); gl = QGridLayout(g2); gl.setSpacing(2); gl.setContentsMargins(4,8,4,4)
        gl.addWidget(QLabel("音频驱动"), 0, 0)
        self.ha_combo = QComboBox(); self.ha_combo.currentTextChanged.connect(self._ha_changed); gl.addWidget(self.ha_combo, 0, 1)
        self.wasapi_chk = QCheckBox("独占设备"); gl.addWidget(self.wasapi_chk, 0, 2)
        gl.addWidget(QLabel("麦克风"), 1, 0)
        self.in_combo = QComboBox(); gl.addWidget(self.in_combo, 1, 1, 1, 2)
        gl.addWidget(QLabel("主输出"), 2, 0)
        self.out_combo = QComboBox(); gl.addWidget(self.out_combo, 2, 1, 1, 2)
        gl.addWidget(QLabel("副输出"), 3, 0)
        self.out2_chk = QCheckBox("启用"); gl.addWidget(self.out2_chk, 3, 1)
        self.out2_combo = QComboBox(); gl.addWidget(self.out2_combo, 3, 2)
        br = QPushButton("刷新列表"); br.clicked.connect(self._reload_dev); gl.addWidget(br, 4, 0)
        self.sr_r1 = QRadioButton("模型采样率"); self.sr_r1.setChecked(True)
        self.sr_r2 = QRadioButton("设备采样率")
        gl.addWidget(self.sr_r1, 4, 1); gl.addWidget(self.sr_r2, 4, 2)
        self.sr_lbl = QLabel("频率: -"); gl.addWidget(self.sr_lbl, 4, 3)
        l.addWidget(g2)
        return w

    def _tab_engine(self):
        w = QWidget(); row = QHBoxLayout(w); row.setSpacing(4)

        # 左列：常规
        left = QGroupBox("常规设置"); lg = QGridLayout(left); lg.setSpacing(2); lg.setContentsMargins(4,8,4,4)
        def add_slider(label, sl, lbl, r):
            lg.addWidget(QLabel(label), r, 0); lg.addWidget(sl, r, 1); lg.addWidget(lbl, r, 2)
        self.th_sl = self._sl(-60,0,1,-60); self.th_lbl = QLabel("-60")
        self.th_sl.valueChanged.connect(lambda v: (self.th_lbl.setText(str(v)), setattr(p,'threshold',v)))
        add_slider("响应阈值", self.th_sl, self.th_lbl, 0)
        self.pit_sl = self._sl(-16,16,1,0); self.pit_lbl = QLabel("0")
        self.pit_sl.valueChanged.connect(lambda v: self.pit_lbl.setText(str(v)))
        add_slider("音调调整", self.pit_sl, self.pit_lbl, 1)
        self.fmt_sl = self._sl(-200,200,5,0); self.fmt_lbl = QLabel("0.00")
        self.fmt_sl.valueChanged.connect(lambda v: self.fmt_lbl.setText(f"{v/100:.2f}"))
        add_slider("声线粗细", self.fmt_sl, self.fmt_lbl, 2)
        self.ir_sl = self._sl(0,100,1,0); self.ir_lbl = QLabel("0.00")
        self.ir_sl.valueChanged.connect(lambda v: self.ir_lbl.setText(f"{v/100:.2f}"))
        add_slider("特征占比", self.ir_sl, self.ir_lbl, 3)
        self.rms_sl = self._sl(0,100,1,0); self.rms_lbl = QLabel("0.00")
        self.rms_sl.valueChanged.connect(lambda v: self.rms_lbl.setText(f"{v/100:.2f}"))
        add_slider("响度因子", self.rms_sl, self.rms_lbl, 4)
        lg.addWidget(QLabel("音高算法"), 5, 0)
        self.f0_grp = QButtonGroup(); fr = QHBoxLayout()
        for i, n in enumerate(["fcpe","rmvpe"]):
            rb = QRadioButton(n)
            if i==0: rb.setChecked(True)
            self.f0_grp.addButton(rb, i); fr.addWidget(rb)
        lg.addLayout(fr, 5, 1, 1, 2)
        row.addWidget(left)

        # 右列：性能
        right = QGroupBox("性能设置"); rg = QGridLayout(right); rg.setSpacing(2); rg.setContentsMargins(4,8,4,4)
        def add_s2(label, sl, lbl, r):
            rg.addWidget(QLabel(label), r, 0); rg.addWidget(sl, r, 1); rg.addWidget(lbl, r, 2)
        self.bl_sl = self._sl(2,150,1,25); self.bl_lbl = QLabel("0.25")
        self.bl_sl.valueChanged.connect(lambda v: self.bl_lbl.setText(f"{v/100:.2f}"))
        add_s2("采样长度", self.bl_sl, self.bl_lbl, 0)
        self.cf_sl = self._sl(1,15,1,5); self.cf_lbl = QLabel("0.05")
        self.cf_sl.valueChanged.connect(lambda v: self.cf_lbl.setText(f"{v/100:.2f}"))
        add_s2("淡入淡出", self.cf_sl, self.cf_lbl, 1)
        self.ex_sl = self._sl(5,500,1,250); self.ex_lbl = QLabel("2.50")
        self.ex_sl.valueChanged.connect(lambda v: self.ex_lbl.setText(f"{v/100:.2f}"))
        add_s2("额外时长", self.ex_sl, self.ex_lbl, 2)
        self.inr = QCheckBox("输入降噪"); self.ounr = QCheckBox("输出降噪"); self.pv = QCheckBox("相位声码")
        rg.addWidget(self.inr, 3, 0); rg.addWidget(self.ounr, 3, 1); rg.addWidget(self.pv, 3, 2)
        row.addWidget(right)
        return w

    def _tab_master(self):
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(3); l.setContentsMargins(4,4,4,4)

        g = QGroupBox("母带级音效处理"); gl = QGridLayout(g); gl.setSpacing(2); gl.setContentsMargins(4,8,4,4)
        self.eq_en = QCheckBox("开启后期模块"); self.eq_en.setStyleSheet("font-weight:bold;color:#bb86fc")
        gl.addWidget(self.eq_en, 0, 0)
        gl.addWidget(QLabel("预设:"), 0, 1)
        self.preset_combo = QComboBox(); self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        gl.addWidget(self.preset_combo, 0, 2)

        self.eq_lo = self._sl(-3000,2000,500,0); self.eq_lo_lbl = QLabel("0.0")
        self.eq_lo.valueChanged.connect(lambda v: self.eq_lo_lbl.setText(f"{v/100:.1f}"))
        gl.addWidget(QLabel("低频增益"), 1, 0); gl.addWidget(self.eq_lo, 1, 1); gl.addWidget(self.eq_lo_lbl, 1, 2)
        self.eq_mi = self._sl(-2000,2000,500,0); self.eq_mi_lbl = QLabel("0.0")
        self.eq_mi.valueChanged.connect(lambda v: self.eq_mi_lbl.setText(f"{v/100:.1f}"))
        gl.addWidget(QLabel("中频增益"), 2, 0); gl.addWidget(self.eq_mi, 2, 1); gl.addWidget(self.eq_mi_lbl, 2, 2)
        self.eq_hi = self._sl(-3000,3000,500,0); self.eq_hi_lbl = QLabel("0.0")
        self.eq_hi.valueChanged.connect(lambda v: self.eq_hi_lbl.setText(f"{v/100:.1f}"))
        gl.addWidget(QLabel("高频增益"), 3, 0); gl.addWidget(self.eq_hi, 3, 1); gl.addWidget(self.eq_hi_lbl, 3, 2)
        self.warm_sl = self._sl(0,100,1,0); self.warm_lbl = QLabel("0.00")
        self.warm_sl.valueChanged.connect(lambda v: self.warm_lbl.setText(f"{v/100:.2f}"))
        gl.addWidget(QLabel("电子管饱和"), 4, 0); gl.addWidget(self.warm_sl, 4, 1); gl.addWidget(self.warm_lbl, 4, 2)
        self.comp_sl = self._sl(0,100,5,0); self.comp_lbl = QLabel("0.00")
        self.comp_sl.valueChanged.connect(lambda v: self.comp_lbl.setText(f"{v/100:.2f}"))
        gl.addWidget(QLabel("动态压限"), 5, 0); gl.addWidget(self.comp_sl, 5, 1); gl.addWidget(self.comp_lbl, 5, 2)
        self.rev_sl = self._sl(0,50,1,0); self.rev_lbl = QLabel("0.00")
        self.rev_sl.valueChanged.connect(lambda v: self.rev_lbl.setText(f"{v/100:.2f}"))
        gl.addWidget(QLabel("空间混响"), 6, 0); gl.addWidget(self.rev_sl, 6, 1); gl.addWidget(self.rev_lbl, 6, 2)
        l.addWidget(g)

        g2 = QGroupBox("离线背景音"); g2l = QGridLayout(g2); g2l.setSpacing(2); g2l.setContentsMargins(4,8,4,4)
        self.bgm_en = QCheckBox("开启BGM"); g2l.addWidget(self.bgm_en, 0, 0)
        self.bgm_edit = QLineEdit(); g2l.addWidget(self.bgm_edit, 0, 1)
        bb = QPushButton("选择"); bb.clicked.connect(lambda: self._browse(self.bgm_edit, "音频 (*.wav *.mp3 *.flac)")); g2l.addWidget(bb, 0, 2)
        g2l.addWidget(QLabel("音量"), 1, 0)
        self.bgm_sl = self._sl(0,200,1,50); self.bgm_lbl = QLabel("0.50")
        self.bgm_sl.valueChanged.connect(lambda v: self.bgm_lbl.setText(f"{v/100:.2f}"))
        g2l.addWidget(self.bgm_sl, 1, 1); g2l.addWidget(self.bgm_lbl, 1, 2)
        l.addWidget(g2)
        return w

    def _apply_preset(self, name):
        if name not in PRESETS: return
        pr = PRESETS[name]
        self.eq_lo.setValue(int(pr["eq_low"]*100)); self.eq_mi.setValue(int(pr["eq_mid"]*100)); self.eq_hi.setValue(int(pr["eq_high"]*100))
        self.warm_sl.setValue(int(pr["warmth"]*100)); self.comp_sl.setValue(int(pr["compress"]*100)); self.rev_sl.setValue(int(pr["reverb"]*100))

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path: tgt.setText(path)

    def _load_cfg(self):
        self._reload_dev()
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f: d = json.load(f)
                self.pth_edit.setText(d.get("pth","")); self.idx_edit.setText(d.get("idx",""))
                self.pit_sl.setValue(d.get("pitch",0)); self.fmt_sl.setValue(int(d.get("formant",0)*100))
                self.ir_sl.setValue(int(d.get("ir",0)*100)); self.rms_sl.setValue(int(d.get("rms",0)*100))
                self.th_sl.setValue(d.get("th",-60)); self.bl_sl.setValue(int(d.get("bl",0.25)*100))
                self.cf_sl.setValue(int(d.get("cf",0.05)*100)); self.ex_sl.setValue(int(d.get("ex",2.5)*100))
                self.inr.setChecked(d.get("inr",False)); self.ounr.setChecked(d.get("ounr",False))
                self.pv.setChecked(d.get("pv",False))
                self.eq_en.setChecked(d.get("eq_en",False))
                self.eq_lo.setValue(int(d.get("eq_lo",0)*100)); self.eq_mi.setValue(int(d.get("eq_mi",0)*100))
                self.eq_hi.setValue(int(d.get("eq_hi",0)*100)); self.warm_sl.setValue(int(d.get("warm",0)*100))
                self.comp_sl.setValue(int(d.get("comp",0)*100)); self.rev_sl.setValue(int(d.get("rev",0)*100))
                self.bgm_en.setChecked(d.get("bgm_en",False)); self.bgm_edit.setText(d.get("bgm",""))
                self.bgm_sl.setValue(int(d.get("bgm_v",0.5)*100)); self.out2_chk.setChecked(d.get("out2",False))
                for btn in self.f0_grp.buttons():
                    if btn.text() == d.get("f0","fcpe"): btn.setChecked(True)
                pr = d.get("preset","原声纯净")
                if pr in PRESETS: self.preset_combo.setCurrentText(pr)
            except: pass

    def _save_cfg(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        f0 = next((b.text() for b in self.f0_grp.buttons() if b.isChecked()), "fcpe")
        d = {
            "pth": self.pth_edit.text(), "idx": self.idx_edit.text(),
            "pitch": self.pit_sl.value(), "formant": self.fmt_sl.value()/100,
            "ir": self.ir_sl.value()/100, "rms": self.rms_sl.value()/100,
            "th": self.th_sl.value(), "bl": self.bl_sl.value()/100,
            "cf": self.cf_sl.value()/100, "ex": self.ex_sl.value()/100,
            "f0": f0, "inr": self.inr.isChecked(), "ounr": self.ounr.isChecked(),
            "pv": self.pv.isChecked(), "eq_en": self.eq_en.isChecked(),
            "eq_lo": self.eq_lo.value()/100, "eq_mi": self.eq_mi.value()/100,
            "eq_hi": self.eq_hi.value()/100, "warm": self.warm_sl.value()/100,
            "comp": self.comp_sl.value()/100, "rev": self.rev_sl.value()/100,
            "bgm_en": self.bgm_en.isChecked(), "bgm": self.bgm_edit.text(),
            "bgm_v": self.bgm_sl.value()/100, "out2": self.out2_chk.isChecked(),
            "preset": self.preset_combo.currentText(),
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, ensure_ascii=False)

    def _reload_dev(self):
        self.ha_combo.blockSignals(True); self.ha_combo.clear()
        names, *_ = get_audio_devices(); self.ha_combo.addItems(names)
        self.ha_combo.blockSignals(False); self._ha_changed(self.ha_combo.currentText())

    def _ha_changed(self, name):
        if not name: return
        _, ins, outs, _, _ = get_audio_devices(name)
        self.in_combo.clear(); self.in_combo.addItems(ins)
        self.out_combo.clear(); self.out_combo.addItems(outs)
        self.out2_combo.clear(); self.out2_combo.addItems(outs)

    def _start(self):
        pth = self.pth_edit.text().strip()
        if not pth: QMessageBox.warning(self, "提示", "请选择 .pth 文件"); return
        self.btn_start.setEnabled(False); self.btn_start.setText("加载中...")
        self._lt = LoadThread(pth, self.idx_edit.text().strip(), self.ir_sl.value()/100)
        self._lt.ok.connect(self._on_loaded); self._lt.err.connect(self._on_err)
        self._lt.start()

    def _on_loaded(self, sr):
        try:
            _, _, _, in_idx, out_idx = get_audio_devices(self.ha_combo.currentText())
            # 写入运行时参数
            p.threshold = self.th_sl.value(); p.pitch = self.pit_sl.value()
            p.formant = self.fmt_sl.value()/100; p.index_rate = self.ir_sl.value()/100
            p.rms_mix = self.rms_sl.value()/100; p.f0method = next((b.text() for b in self.f0_grp.buttons() if b.isChecked()), "fcpe")
            p.I_nr = self.inr.isChecked(); p.O_nr = self.ounr.isChecked(); p.use_pv = self.pv.isChecked()
            p.enable_eq = self.eq_en.isChecked(); p.eq_low = self.eq_lo.value()/100
            p.eq_mid = self.eq_mi.value()/100; p.eq_hi = self.eq_hi.value()/100
            p.warmth = self.warm_sl.value()/100; p.compress = self.comp_sl.value()/100
            p.reverb = self.rev_sl.value()/100; p.bgm_enable = self.bgm_en.isChecked()
            p.bgm_vol = self.bgm_sl.value()/100; p.enable_out2 = self.out2_chk.isChecked()
            sr_type = "sr_model" if self.sr_r1.isChecked() else "sr_device"
            engine.setup(sr_type, in_idx[self.in_combo.currentIndex()], out_idx[self.out_combo.currentIndex()],
                         self.wasapi_chk.isChecked(), self.bl_sl.value()/100, self.cf_sl.value()/100, self.ex_sl.value()/100, p)
            # BGM
            engine.bgm_audio = None; engine.bgm_ptr = 0
            if p.bgm_enable and self.bgm_edit.text() and os.path.exists(self.bgm_edit.text()):
                try:
                    w, sr2 = torchaudio.load(self.bgm_edit.text())
                    if w.shape[0]>1: w = torch.mean(w, 0, keepdim=True)
                    if sr2 != engine.sr: w = TatResample(sr2, engine.sr)(w)
                    engine.bgm_audio = w.squeeze(0)
                except Exception as e: logger.warning("BGM加载失败: %s", e)
            # 副输出
            if p.enable_out2 and self.out2_combo.currentIndex() >= 0:
                _, _, _, _, o2i = get_audio_devices(self.ha_combo.currentText())
                engine.setup_out2(o2i[self.out2_combo.currentIndex()])
            self.sr_lbl.setText(f"设备:{engine.sr_dev} 模型:{engine.sr_model}")
            dl = (engine.stream.latency[-1] if engine.stream else 0) + self.bl_sl.value()/100 + self.cf_sl.value()/100 + 0.01
            if p.I_nr: dl += min(self.cf_sl.value()/100, 0.04)
            self.delay_lbl.setText(f"延迟: {int(dl*1000)} ms")
            self.btn_start.setEnabled(False); self._timer.start(200); self._save_cfg()
        except Exception as e: self._on_err(str(e))

    def _on_err(self, e):
        self.btn_start.setEnabled(True); self.btn_start.setText("▶ 开始转换")
        QMessageBox.critical(self, "错误", str(e))

    def _stop(self):
        self._timer.stop(); engine.stop(); self.btn_start.setEnabled(True)
        self.delay_lbl.setText("延迟: - ms"); self.stat_lbl.setText("推理: - ms")

    def closeEvent(self, e):
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
