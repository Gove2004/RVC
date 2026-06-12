from pathlib import Path

import numpy as np

from rvc.audio_loader import load_audio
from rvc.rmvpe import RMVPE

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


class F0Extractor:
    def __init__(self, device: str = "cuda:0", is_half: bool = True):
        self.device = device
        self.is_half = is_half
        self.model = RMVPE("assets/rmvpe/rmvpe.pt", is_half=is_half, device=device)
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self, exp_dir: str, progress_callback=None):
        exp = Path(exp_dir)
        wav_dir = exp / "1_16k_wavs"
        coarse_dir = exp / "2a_f0"
        continuous_dir = exp / "2b-f0nsf"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        continuous_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(wav_dir.glob("*.wav"))
        for i, path in enumerate(files, 1):
            if self.stop_requested:
                break
            out_coarse = coarse_dir / f"{path.stem}.npy"
            out_cont = continuous_dir / f"{path.stem}.npy"
            if not out_coarse.exists() or not out_cont.exists():
                wav, _ = load_audio(path, 16000)
                f0 = self.model.infer_from_audio(wav, thred=0.03)
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
