from rvc.core.constants import HUBERT_SAMPLE_RATE
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rvc.io.audio_file import load_audio
from rvc.models.hubert import load_hubert
from rvc.runtime.caches import LRUCache

# 训练侧 HuBERT 进程级缓存（与推理侧 InferenceCache 的 hubert 槽位同配置 LRU=2）：
# 同进程第二次训练任务直接复用，避免重新加载 ~1GB 权重。
# 缓存策略在本层（调用方）——模型层 load_hubert 是纯加载器。
_hubert_cache = LRUCache(2)


class HuBERTExtractor:
    def __init__(self, device: str = "cuda:0", is_half: bool = True, hubert: str = "base",
                 ffmpeg_exe: str | None = None):
        self.device = device
        self.is_half = is_half
        self.hubert = hubert
        self.ffmpeg_exe = ffmpeg_exe
        cache_key = (device, is_half, hubert)
        model = _hubert_cache.get(cache_key)
        if model is None:
            model = load_hubert(SimpleNamespace(device=device, is_half=is_half), variant=hubert)
            _hubert_cache.set(cache_key, model)
        self.model = model
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
        wav, _ = load_audio(path, HUBERT_SAMPLE_RATE, ffmpeg_exe=self.ffmpeg_exe)
        feats = torch.from_numpy(wav).to(self.device)
        feats = feats.half() if self.is_half else feats.float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            # transformers 模型返回 BaseModelOutput，取 last_hidden_state（与推理侧一致）
            feats_result = self.model(feats).last_hidden_state
            feats = feats_result.squeeze(0).float().cpu().numpy()
        return feats.astype(np.float32)
