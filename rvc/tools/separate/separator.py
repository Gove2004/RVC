"""人声提纯高层 API。

链路：加载音频 → 依次跑一串模型（每级取自己那一路 stem）→ 转单声道写 wav。
所有模型都在 44.1kHz 立体声上工作，长音频由 demix 的分块滑窗处理，无时长上限。

注意 VR 架构输出是 (N, 2)，demix 输出是 (2, N)，这里统一成 (2, N) 再往下传。
"""
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.audio.loader import load_audio
from . import registry
from .checkpoint import load_model_weights
from .demix import demix, get_model_from_config

logger = logging.getLogger(__name__)

MODEL_SR = 44100

# VR 架构推理参数（沿用 UVR 默认值，够用且稳）
_VR_ARCH_CONFIG = {
    "batch_size": 2,
    "window_size": 512,
    "aggression": 5,
    "enable_tta": False,
    "enable_post_process": False,
    "post_process_threshold": 0.2,
    "high_end_process": False,
    "use_amp": True,
    "fuse_conv_bn": False,
    "use_channels_last": True,
}


def _as_2ch(wav: np.ndarray) -> np.ndarray:
    """统一成 (2, N) float32 立体声。"""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        return np.asfortranarray(np.stack([wav, wav]))
    if wav.shape[0] == 2:
        return np.asfortranarray(wav)
    if wav.shape[1] == 2:  # VR 输出的 (N, 2)
        return np.asfortranarray(wav.T)
    return np.asfortranarray(np.stack([wav[0], wav[0]]))


def _pick_stem(sources: dict, stem: str, key: str) -> np.ndarray:
    """从模型输出里取出要保留的那一路。"""
    if stem in sources:
        return sources[stem]
    # 大小写/空格差异兜底
    normalized = {str(k).strip().lower(): v for k, v in sources.items()}
    wanted = stem.strip().lower()
    if wanted in normalized:
        return normalized[wanted]
    if len(sources) == 1:
        return next(iter(sources.values()))
    raise KeyError(f"模型 {key} 的输出里找不到 stem {stem!r}，实际输出：{list(sources)}")


class VocalSeparator:
    def __init__(self, device=None):
        from rvc.runtime import Config

        config = Config()
        self.device = device or config.device
        self._loaded: dict[str, object] = {}

    # ── 模型管理 ──────────────────────────────────────────

    def load(self, key: str):
        """惰性加载并缓存模型；重复调用直接返回缓存。"""
        if key in self._loaded:
            return self._loaded[key]

        spec = registry.get(key)
        if not spec.is_ready:
            raise FileNotFoundError(f"模型未下载：{spec.label}\n期望路径：{spec.weight_path}")

        if spec.model_type == "vr":
            from .modules.vocal_remover.vr_models import get_vr_model_metadata
            from .modules.vocal_remover.vr_separator import VRSeparator

            model_data = get_vr_model_metadata(str(spec.weight_path))
            common_config = {
                "logger": logger,
                "torch_device": str(self.device),
                "torch_device_cpu": "cpu",
                "torch_device_mps": None,
                "model_name": model_data["model_name"],
                "model_path": str(spec.weight_path),
                "model_data": model_data,
                "sample_rate": MODEL_SR,
            }
            separator = VRSeparator(common_config, dict(_VR_ARCH_CONFIG))
            separator.load_model()
            entry = separator
        else:
            model, config = get_model_from_config(spec.model_type, str(spec.config_path))
            load_model_weights(model, spec.weight_path, model_type=spec.model_type, map_location="cpu")
            model.to(self.device).eval()
            entry = (model, config)

        self._loaded[key] = entry
        return entry

    def unload(self, key: str):
        """释放单个模型显存。"""
        entry = self._loaded.pop(key, None)
        if entry is None:
            return
        if isinstance(entry, tuple):
            del entry
        torch.cuda.empty_cache()

    def unload_all(self):
        for key in list(self._loaded):
            self.unload(key)

    # ── 推理 ──────────────────────────────────────────────

    def separate_array(self, mix: np.ndarray, keys, progress=None, stop_check=None) -> np.ndarray:
        """对 (2, N) @44.1k 的音频依次跑 keys 里的模型，返回提纯后的 (2, N)。"""
        current = _as_2ch(mix)
        total = len(keys)
        for index, key in enumerate(keys, 1):
            if stop_check and stop_check():
                raise RuntimeError("已取消")
            spec = registry.get(key)
            entry = self.load(key)

            if spec.model_type == "vr":
                sources = entry.separate_array(current, MODEL_SR)
                picked = _as_2ch(_pick_stem(sources, spec.stem, key))
                # VR 走频谱往返，iSTFT 后长度有 hop 量化误差（几毫秒），统一对齐回输入长度
                if picked.shape[1] != current.shape[1]:
                    if picked.shape[1] > current.shape[1]:
                        picked = picked[:, : current.shape[1]]
                    else:
                        picked = np.pad(picked, ((0, 0), (0, current.shape[1] - picked.shape[1])), mode="edge")
                current = picked
            else:
                model, config = entry
                sources = demix(
                    config,
                    model,
                    current,
                    self.device,
                    model_type=spec.model_type,
                    progress_callback=self._make_stage_progress(progress, index, total),
                )
                current = _as_2ch(_pick_stem(sources, spec.stem, key))

            if np.isnan(current).any():
                np.nan_to_num(current, copy=False, nan=0.0)
        return current

    @staticmethod
    def _make_stage_progress(progress, index, total):
        """把单级 0~1 进度映射到整条链路的 [index-1, index] 区间。

        签名是上游 _ProgressContext 的三参数 (done, total, message)。
        """
        if progress is None:
            return None

        def _emit(position, total_position, message=None):
            fraction = position / max(total_position, 1)
            progress(int((index - 1 + fraction) * 1000), total * 1000)

        return _emit

    # ── 文件级处理 ────────────────────────────────────────

    def process_file(self, in_path: Path, out_path: Path, keys, out_sr: int = MODEL_SR,
                     progress=None, stop_check=None) -> Path:
        """处理单个音频文件，返回输出路径。"""
        wav, _ = load_audio(in_path, MODEL_SR, mono=False)
        if wav.size == 0 or float(np.abs(wav).max()) < 1e-6:
            raise ValueError(f"音频为空或几乎静音：{in_path.name}")

        result = self.separate_array(wav, keys, progress=progress, stop_check=stop_check)

        mono = result.mean(axis=0, dtype=np.float32)
        if out_sr != MODEL_SR:
            import librosa

            mono = librosa.resample(mono, orig_sr=MODEL_SR, target_sr=int(out_sr)).astype(np.float32)
        peak = float(np.abs(mono).max())
        if peak > 0:
            mono = (mono / peak * 0.95).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), mono, int(out_sr), subtype="PCM_16")  # 16bit 通用且体积小一半
        return out_path
