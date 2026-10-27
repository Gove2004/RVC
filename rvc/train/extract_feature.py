"""训练侧 HuBERT 特征提取 — 批量把 16k wav 转为 768 维内容特征 .npy。

HuBERTExtractor 带进程级缓存（同配置模型只加载一次），供训练向导复用。
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rvc.core.constants import HUBERT_SAMPLE_RATE
from rvc.io.audio_file import load_audio
from rvc.models.hubert import load_hubert
from rvc.pipeline.features import hubert_forward
from rvc.runtime.graph import purge_cuda_graphs
from rvc.runtime.lru import LRUCache
from rvc.train.atomic_save import save_npy_atomic

# 训练侧 HuBERT 进程级缓存（与推理侧 InferenceCache 的 hubert 槽位同配置 LRU=2）：
# 同进程第二次训练任务直接复用，避免重新加载 ~1GB 权重。
# 缓存策略在本层（调用方）——模型层 load_hubert 是纯加载器。
_hubert_cache = LRUCache(2)


def clear_hubert_cache():
    """清空训练侧 HuBERT 缓存并归还显存（特征提取完成后调用，训练前释放 ~1GB）。"""
    # LRUCache.clear() 只丢引用，不回收值本身；HuBERT 可能持有 CUDA Graph 缓存，
    # 其静态输入/输出张量不会随引用消失立刻归还。先逐个回收再清空。
    for model in _hubert_cache.values():
        purge_cuda_graphs(model)
    _hubert_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
            # 淘汰即回收：被淘汰的 HuBERT 可能持有 CUDA Graph 缓存（从 GUI 与推理侧
            # 同进程跑训练时图是启用的），其静态输入/输出张量占的显存不会随引用消失
            # 立刻归还，而新模型捕获图时可能复用同一批地址。回收动作就放在这一行。
            evicted = _hubert_cache.insert(cache_key, model)
            if evicted is not None:
                purge_cuda_graphs(evicted)
        self.model = model

    def run(self, exp_dir: str, progress_callback=None, stop_check=None):
        exp = Path(exp_dir)
        wav_dir = exp / "1_16k_wavs"
        feat_dir = exp / "3_feature768"
        feat_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(wav_dir.glob("*.wav"))
        done = 0
        for i, path in enumerate(files, 1):
            if stop_check is not None and stop_check():
                break
            done = i
            out_path = feat_dir / f"{path.stem}.npy"
            if not out_path.exists():
                feats = self.extract(path)
                # 原子写：中途强杀不留截断 npy 被续跑逻辑静默复用
                save_npy_atomic(out_path, feats)
            if progress_callback:
                progress_callback(i, len(files))
        # 返回实际完成的条数（中止时 ≠ 全量，调用方据此打准确的完成日志）
        return done

    def extract(self, path: Path):
        wav, _ = load_audio(path, HUBERT_SAMPLE_RATE, ffmpeg_exe=self.ffmpeg_exe)
        feats = torch.from_numpy(wav).to(self.device)
        feats = feats.half() if self.is_half else feats.float()
        feats = feats.view(1, -1)
        with torch.inference_mode():
            # 与推理侧共享 hubert_forward，保证模型前向传播路径一致
            feats_result = hubert_forward(self.model, feats)
            feats = feats_result.squeeze(0).float().cpu().numpy()
        return feats.astype(np.float32)
