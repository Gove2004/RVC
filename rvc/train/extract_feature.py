from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rvc.audio_loader import load_audio
from rvc.hubert import load_hubert


class HuBERTExtractor:
    def __init__(self, device: str = "cuda:0", is_half: bool = True):
        self.device = device
        self.is_half = is_half
        self.model = load_hubert(SimpleNamespace(device=device, is_half=is_half))

    def run(self, exp_dir: str, progress_callback=None):
        exp = Path(exp_dir)
        wav_dir = exp / "1_16k_wavs"
        feat_dir = exp / "3_feature768"
        feat_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(wav_dir.glob("*.wav"))
        for i, path in enumerate(files, 1):
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
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model.extract_features(source=feats, padding_mask=padding_mask, output_layer=12)
            feats = logits[0].squeeze(0).float().cpu().numpy()
        return feats.astype(np.float32)
