from pathlib import Path

import numpy as np

from rvc.audio.loader import load_audio
from rvc.inference.f0_extractor import F0_MEL_MAX, F0_MEL_MIN  # 复用推理侧常量，避免算法漂移
from rvc.models.rmvpe import RMVPE
from rvc.runtime.paths import RMVPE_PATH


class TrainF0Extractor:
    """训练用 F0 提取器（批量处理切片 wav → 离散/连续 F0 npy）。

    与推理侧 `rvc.inference.f0_extractor.F0Extractor`（抽象基类，RMVPE/FCPE 二选一）
    职责不同，名字刻意区分，避免混淆。
    """

    def __init__(self, device: str = "cuda:0", is_half: bool = True):
        self.device = device
        self.is_half = is_half
        self.model = RMVPE(str(RMVPE_PATH), is_half=is_half, device=device)
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self, exp_dir: str, progress_callback=None, stop_check=None):
        exp = Path(exp_dir)
        wav_dir = exp / "1_16k_wavs"
        coarse_dir = exp / "2a_f0"
        continuous_dir = exp / "2b-f0nsf"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        continuous_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(wav_dir.glob("*.wav"))
        for i, path in enumerate(files, 1):
            if self.stop_requested or (stop_check is not None and stop_check()):
                break
            out_coarse = coarse_dir / f"{path.stem}.npy"
            out_cont = continuous_dir / f"{path.stem}.npy"
            if not out_coarse.exists() or not out_cont.exists():
                wav, _ = load_audio(path, 16000)
                f0 = self.model.infer_from_audio(wav, thred=0.03)
                # 推理侧解码已搬上 GPU，训练侧要落盘 npy 才转回 CPU
                f0 = f0.detach().float().cpu().numpy()
                np.save(out_cont, f0.astype(np.float32), allow_pickle=False)
                np.save(out_coarse, coarse_f0(f0), allow_pickle=False)
            if progress_callback:
                progress_callback(i, len(files))
        return len(files)


def coarse_f0(f0: np.ndarray):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)
