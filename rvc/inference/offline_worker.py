"""离线推理 — 音频文件转换单次推理"""
import logging
import traceback

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from gui.configs import Config
from rvc.audio.loader import load_audio_native
from rvc.audio.effects import create_offline_chain
from rvc.audio.utils import match_rms

logger = logging.getLogger(__name__)
config = Config()

_X_PAD = 3  # 与 Config 中 x_pad 一致


class OfflineWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_path, output_path, pth, idx, idx_rate, pitch, f0method, rms_mix, protect,
                 enable_eq=False, eq_bands=None, reverb_mix=0.0):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.pth = pth; self.idx = idx; self.idx_rate = idx_rate
        self.pitch = pitch; self.f0method = f0method; self.rms_mix = rms_mix
        self.protect = protect

        # 声学参数
        self.enable_eq = enable_eq
        self.eq_bands = eq_bands or {'sub': 0, 'low': 0, 'mid': 0, 'hi_mid': 0, 'high': 0}
        self.reverb_mix = reverb_mix

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
        from rvc.inference.pipeline import VCPipeline
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
        self.progress.emit(75, 100)

        # Trim padding
        audio1 = audio1[t_pad_tgt : -t_pad_tgt] if t_pad_tgt > 0 else audio1

        # RMS 响度匹配
        if self.rms_mix != 1:
            audio1 = match_rms(wav, 16000, audio1, tgt_sr, self.rms_mix)

        # 声学效果处理
        if self.enable_eq:
            self.progress.emit(85, 100)
            audio1 = self._apply_effects(audio1, tgt_sr)

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

    def _apply_effects(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """应用声学效果链（离线模式）"""
        # 创建离线效果链
        chain, eq, reverb = create_offline_chain(sr)

        # 配置参数
        eq.set_band('sub', self.eq_bands['sub'])
        eq.set_band('low', self.eq_bands['low'])
        eq.set_band('mid', self.eq_bands['mid'])
        eq.set_band('hi_mid', self.eq_bands['hi_mid'])
        eq.set_band('high', self.eq_bands['high'])
        reverb.set_mix(self.reverb_mix)

        # 转换为 Tensor 处理
        audio_t = torch.from_numpy(audio).to(config.device)
        audio_t = chain(audio_t)

        return audio_t.cpu().numpy()
