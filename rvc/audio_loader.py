"""统一音频加载工具 — 支持任意格式，自动 fallback 到 ffmpeg"""
import logging
import re
import subprocess
import warnings
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FFMPEG = _PROJECT_ROOT / "assets" / "ffmpeg" / "ffmpeg.exe"


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    """加载任意格式音频，自动 fallback 到 ffmpeg 解码。

    Returns:
        (wav_array, sample_rate) — wav 为 float32，sample_rate == target_sr
    """
    path = Path(path).resolve()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*PySoundFile.*")
            warnings.filterwarnings("ignore", message=".*audioread.*", category=FutureWarning)
            wav, sr = librosa.load(str(path), sr=target_sr, mono=mono)
            return wav.astype(np.float32), sr
    except Exception:
        pass
    return _load_via_ffmpeg(path, target_sr)


def load_audio_native(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """加载音频并保持原始采样率（不做重采样）。

    Returns:
        (wav_array, native_sample_rate)
    """
    path = Path(path).resolve()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*PySoundFile.*")
            warnings.filterwarnings("ignore", message=".*audioread.*", category=FutureWarning)
            wav, sr = librosa.load(str(path), sr=None, mono=mono)
            return wav.astype(np.float32), sr
    except Exception:
        pass
    return _load_via_ffmpeg_native(path)


def _load_via_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """通过 ffmpeg 解码并重采样到 target_sr。"""
    if not _FFMPEG.exists():
        raise FileNotFoundError(f"找不到 ffmpeg: {_FFMPEG}\n也无法用 librosa 加载: {path}")
    cmd = [
        str(_FFMPEG), "-i", str(path), "-vn",
        "-acodec", "pcm_f32le", "-f", "f32le",
        "-ac", "1", "-ar", str(target_sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg 解码失败: {path}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    return raw.astype(np.float32), target_sr


def _load_via_ffmpeg_native(path: Path) -> tuple[np.ndarray, int]:
    """通过 ffmpeg 解码，保持原始采样率。"""
    if not _FFMPEG.exists():
        raise FileNotFoundError(f"找不到 ffmpeg: {_FFMPEG}\n也无法用 librosa 加载: {path}")
    # 探测采样率
    info = subprocess.run([str(_FFMPEG), "-i", str(path)], capture_output=True, text=True)
    sr = 48000
    for line in info.stderr.split("\n"):
        if "Hz" in line and "Audio" in line:
            m = re.search(r"(\d+) Hz", line)
            if m:
                sr = int(m.group(1))
                break
    cmd = [
        str(_FFMPEG), "-i", str(path), "-vn",
        "-acodec", "pcm_f32le", "-f", "f32le", "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg 解码失败: {path}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    return raw.astype(np.float32), sr
