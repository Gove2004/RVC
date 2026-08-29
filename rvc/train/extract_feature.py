from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rvc.audio.loader import load_audio
from rvc.models.hubert import load_hubert


class HuBERTExtractor:
    def __init__(self, device: str = "cuda:0", is_half: bool = True):
        self.device = device
        self.is_half = is_half
        self.model = load_hubert(SimpleNamespace(device=device, is_half=is_half))
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self, exp_dir: str, progress_callback=None, stop_check=None):
        exp = Path(exp_dir)
        wav_dir = exp / "1_16k_wavs"
        feat_dir = exp / "3_feature768"
        feat_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(wav_dir.glob("*.wav"))
        for i, path in enumerate(files, 1):
            if self.stop_requested or (stop_check is not None and stop_check()):
                break
            out_path = feat_dir / f"{path.stem}.npy"
            if not out_path.exists():
                feats = self.extract(path)
                np.save(out_path, feats, allow_pickle=False)
            if progress_callback:
                progress_callback(i, len(files))
        return len(files)

    def extract(self, path: Path):
        wav, _ = load_audio(path, 16000)
        feats = torch.from_numpy(wav).to(self.device)
        feats = feats.half() if self.is_half else feats.float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            # transformers 模型返回 BaseModelOutput，取 last_hidden_state（与推理侧一致）
            feats_result = self.model(feats)
            feats_result = getattr(feats_result, "last_hidden_state", feats_result)
            feats = feats_result.squeeze(0).float().cpu().numpy()
        return feats.astype(np.float32)
