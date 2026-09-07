"""音频数据加载器 — 基于 ffmpeg 解码任意格式到 numpy 数组。

职责：把磁盘上的音频文件（mp3/flac/ogg/wav/m4a 等）解码并重采样为
float32 numpy 数组。这是"数据加载"，不是"文件读写"。

与 wav_io 的分工：
- loader.py: 解码任意格式 → numpy 数组（需要 ffmpeg）
- wav_io.py: WAV 文件结构化读写（写 FLOAT32/PCM16，读元信息）
"""
import logging
from rvc.core.errors import AudioLoadError
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


def load_audio(path: str | Path, target_sr: int, mono: bool = True, timeout: int = 300) -> tuple[np.ndarray, int]:
    """用 ffmpeg 解码并重采样到 target_sr，输出 float32。

    mono=False 时返回 shape (2, N) 的立体声。
    timeout: ffmpeg 子进程超时秒数（默认 300，长音频按需调大）。
    """
    path = Path(path).resolve()
    cmd = [
        str(_ffmpeg()), "-i", str(path), "-vn",
        "-acodec", "pcm_f32le", "-f", "f32le",
        "-ac", "1" if mono else "2", "-ar", str(target_sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioLoadError(f"ffmpeg 解码失败: {path}\n{err}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        raw = raw.reshape(-1, 2).T  # f32le 交织 LRLR… → (2, N)
    return raw.astype(np.float32), target_sr
