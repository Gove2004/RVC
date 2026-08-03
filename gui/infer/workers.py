"""推理 GUI 线程 worker。"""
import logging
import traceback

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from rvc.runtime import Config
from rvc.audio.loader import load_audio_native
from rvc.audio.effects import create_offline_chain
from rvc.audio.utils import match_rms
from rvc.inference.offline_config import OfflineConfig

logger = logging.getLogger(__name__)
config = Config()
AUDIO_PAD_SECONDS = 3
MAX_AUDIO_DURATION = 300


class OfflineWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, cfg: OfflineConfig):
        super().__init__()
        self.cfg = cfg

    def run(self):
        vc = None
        try:
            self._do_run()
        except Exception:
            tb = traceback.format_exc()
            logger.error("离线推理失败:\n%s", tb)
            self.error.emit(tb.strip())
        finally:
            try:
                if vc is not None:
                    del vc
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _do_run(self):
        import soundfile as sf

        self.progress.emit(0, 100)
        wav, sr = load_audio_native(self.cfg.input_path)
        duration = len(wav) / sr
        if duration > MAX_AUDIO_DURATION:
            self.error.emit(f"音频时长 {duration:.0f}s 超过限制（最长 {MAX_AUDIO_DURATION // 60} 分钟）")
            self.finished.emit("")
            return

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        self.progress.emit(10, 100)

        from rvc.inference.pipeline import VCPipeline
        from rvc.runtime import Config  # 需要导入 runtime 的 Config
        
        # 使用运行时配置（device, is_half 等）创建 pipeline
        vc = VCPipeline(Config(), self.cfg.model_path, self.cfg.index_path, self.cfg.index_rate)
        vc.load()
        # 根据 gender 设置 formant shift（与实时推理保持一致）
        gender = getattr(self.cfg, 'gender', 0.0)  # 兼容旧版 config 没有 gender 的情况（默认 0.0）
        vc.change_formant(gender)  # 此时 gender 已是 [-2.5, 2.5] 的 formant_shift 值
        vc.change_key(self.cfg.pitch)
        self.progress.emit(20, 100)

        tgt_sr = vc.target_sr
        t_pad = 16000 * AUDIO_PAD_SECONDS
        t_pad_tgt = tgt_sr * AUDIO_PAD_SECONDS
        audio_pad = np.pad(wav, (t_pad, t_pad), mode="reflect")
        self.progress.emit(40, 100)

        audio1 = vc.infer_offline(audio_pad, self.cfg.f0method, self.cfg.protect)
        self.progress.emit(75, 100)
        audio1 = audio1[t_pad_tgt : -t_pad_tgt] if t_pad_tgt > 0 else audio1

        if self.cfg.rms_mix != 1:
            audio1 = match_rms(wav, 16000, audio1, tgt_sr, self.cfg.rms_mix)

        if self.cfg.eq_enabled:
            self.progress.emit(85, 100)
            audio1 = self._apply_effects(audio1, tgt_sr)

        audio_max = np.abs(audio1).max() / 0.99
        if audio_max > 1:
            audio1 = audio1 / audio_max
        sf.write(self.cfg.output_path, audio1, tgt_sr, subtype="FLOAT")
        self.progress.emit(100, 100)
        self.finished.emit(self.cfg.output_path)

    def _apply_effects(self, audio: np.ndarray, sr: int) -> np.ndarray:
        chain, eq, reverb = create_offline_chain(sr)
        eq.set_band('sub', self.cfg.eq_bands['sub'])
        eq.set_band('low', self.cfg.eq_bands['low'])
        eq.set_band('mid', self.cfg.eq_bands['mid'])
        eq.set_band('hi_mid', self.cfg.eq_bands['hi_mid'])
        eq.set_band('high', self.cfg.eq_bands['high'])
        reverb.set_mix(self.cfg.reverb_mix)
        audio_t = torch.from_numpy(audio).to(config.device)
        audio_t = chain(audio_t)
        return audio_t.cpu().numpy()
