"""离线推理 — 音频文件转换单次推理"""
import logging
import os
import traceback

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from configs.config import Config

logger = logging.getLogger(__name__)
config = Config()

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
        import soundfile as sf

        # 加载音频
        self.progress.emit(0, 100)
        wav, sr = self._load_audio(self.input_path)

        # 时长限制
        duration = len(wav) / sr
        if duration > 60:
            self.error.emit(f"音频时长 {duration:.0f}s 超过限制（最长 60 秒）")
            return

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
            pitch_coarse = pitch_coarse[:p_len].unsqueeze(0).contiguous()
            pitchf = pitchf[:p_len].unsqueeze(0).contiguous()
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
