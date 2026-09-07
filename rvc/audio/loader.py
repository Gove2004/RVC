"""统一音频加载工具 — 基于 ffmpeg 解码。"""
import logging
import subprocess
from pathlib import Path

import numpy as np

from rvc.runtime.paths import FFMPEG_EXE

logger = logging.getLogger(__name__)


def _ffmpeg() -> Path:
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(
            f"找不到 ffmpeg: {FFMPEG_EXE}\n"
            f"请将 ffmpeg.exe 放到 assets/ffmpeg/ 目录下。"
        )
    return FFMPEG_EXE


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    """用 ffmpeg 解码并重采样到 target_sr，输出 float32。

    mono=False 时返回 shape (2, N) 的立体声。
    """
    path = Path(path).resolve()
    cmd = [
        str(_ffmpeg()), "-i", str(path), "-vn",
        "-acodec", "pcm_f32le", "-f", "f32le",
        "-ac", "1" if mono else "2", "-ar", str(target_sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg 解码失败: {path}\n{err}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        raw = raw.reshape(-1, 2).T  # f32le 交织 LRLR… → (2, N)
    return raw.astype(np.float32), target_sr
