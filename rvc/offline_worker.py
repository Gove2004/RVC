"""离线推理 — 音频文件转换单次推理"""
import logging
import traceback

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from configs.config import Config
from rvc.audio_loader import load_audio_native

logger = logging.getLogger(__name__)
config = Config()

_X_PAD = 3  # 与 Config 中 x_pad 一致


class OfflineWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_path, output_path, pth, idx, idx_rate, pitch, f0method, rms_mix, protect):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.pth = pth; self.idx = idx; self.idx_rate = idx_rate
        self.pitch = pitch; self.f0method = f0method; self.rms_mix = rms_mix
        self.protect = protect

    def run(self):
        try:
            self._do_run()
        except Exception:
            tb = traceback.format_exc()
            logger.error("离线推理失败:\n%s", tb)
            self.error.emit(tb.strip())

    def _do_run(self):
        import soundfile as sf

        # 加载音频
        self.progress.emit(0, 100)
        wav, sr = load_audio_native(self.input_path)

        # 时长限制
        duration = len(wav) / sr
        if duration > 300:
            self.error.emit(f"音频时长 {duration:.0f}s 超过限制（最长 5 分钟）")
            self.finished.emit("")
            return

        # 重采样到 16kHz（HuBERT 要求）
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        self.progress.emit(10, 100)

        # 推理引擎
        from rvc.vc_pipeline import VCPipeline
        vc = VCPipeline(config, self.pth, self.idx, self.idx_rate)
        vc.load()
        vc.change_key(self.pitch)
        self.progress.emit(20, 100)

        tgt_sr = vc.tgt_sr
        t_pad = 16000 * _X_PAD  # 48000 样本 (16kHz 下 3s)
        t_pad_tgt = tgt_sr * _X_PAD  # 模型采样率下的 padding

        # Pad 音频（反射填充）
        audio_pad = np.pad(wav, (t_pad, t_pad), mode="reflect")
        self.progress.emit(40, 100)

        audio1 = vc.infer_offline(audio_pad, self.f0method, self.protect)
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

        # 释放 GPU 显存
        del vc
        torch.cuda.empty_cache()

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
