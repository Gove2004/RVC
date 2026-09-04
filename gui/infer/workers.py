"""推理 GUI 线程 worker。"""
import logging
import traceback

from PySide6.QtCore import QThread, Signal

from rvc.inference.offline_config import OfflineConfig

logger = logging.getLogger(__name__)
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
        import torch  # 惰性导入，避免 GUI 启动时加载 torch

        try:
            self._do_run()
        except Exception:
            tb = traceback.format_exc()
            logger.error("离线推理失败:\n%s", tb)
            self.error.emit(tb.strip())
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _do_run(self):
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        from rvc.audio.loader import load_audio_native
        from rvc.audio.realtime_mix import apply_rms_mix

        self.progress.emit(0, 100)
        wav, sr = load_audio_native(self.cfg.input_path)
        duration = len(wav) / sr
        if duration > MAX_AUDIO_DURATION:
            # 只发 error 不发 finished，避免 UI 状态被「完成」覆盖
            self.error.emit(f"音频时长 {duration:.0f}s 超过限制（最长 {MAX_AUDIO_DURATION // 60} 分钟）")
            return

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        self.progress.emit(10, 100)

        from rvc.inference.pipeline import VCPipeline
        from rvc.runtime import Config  # 需要导入 runtime 的 Config
        
        # 使用运行时配置（device, is_half 等）创建 pipeline
        vc = VCPipeline(Config(), self.cfg.model_path, hubert=self.cfg.hubert)
        vc.load()
        # 与实时路径一致的参数应用（统一走 VCPipeline.configure，保证实时/离线同源）
        vc.configure(
            pitch=self.cfg.pitch,
            gender=self.cfg.gender,
            break_enable=self.cfg.break_enable,
            break_src_hz=self.cfg.break_src_hz,
        )
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
            # 统一走 torch GPU 版 RMS（与实时路径同实现，ref_hz=160 因源 16k ≠ 目标 sr）
            ref = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
            conv = torch.from_numpy(np.ascontiguousarray(audio1, dtype=np.float32))
            audio1 = apply_rms_mix(ref, conv, self.cfg.rms_mix, tgt_sr // 100, ref_hz=160).numpy()

        audio_max = np.abs(audio1).max() / 0.99
        if audio_max > 1:
            audio1 = audio1 / audio_max
        sf.write(self.cfg.output_path, audio1, tgt_sr, subtype="FLOAT")
        self.progress.emit(100, 100)
        self.finished.emit(self.cfg.output_path)
