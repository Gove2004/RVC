"""轻量级 WAV 读写（替代 soundfile）。

仅支持项目需要的格式：
- 写：FLOAT32（IEEE float）、PCM16
- 读元信息：WAV 直接解析头；其他格式用 ffmpeg 解析
"""
import re
import struct
import subprocess
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, data: np.ndarray, samplerate: int, subtype: str = "FLOAT"):
    """写 WAV 文件。

    Args:
        data: 1D（单声道）或 2D（frames, channels）数组，取值范围 [-1, 1]
        samplerate: 采样率
        subtype: "FLOAT"（32-bit IEEE float）或 "PCM_16"（16-bit 整数）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    frames, channels = data.shape

    if subtype == "FLOAT":
        audio_format = 3  # WAVE_FORMAT_IEEE_FLOAT
        bits_per_sample = 32
        raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
    elif subtype == "PCM_16":
        audio_format = 1  # WAVE_FORMAT_PCM
        bits_per_sample = 16
        clipped = np.clip(data, -1.0, 1.0)
        raw = np.ascontiguousarray(clipped * 32767, dtype=np.int16).tobytes()
    else:
        raise ValueError(f"不支持的 subtype: {subtype}（仅支持 FLOAT / PCM_16）")

    byte_rate = samplerate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(raw)

    # RIFF header (12 bytes)
    riff_size = 36 + data_size
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")

    # fmt chunk (24 bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16, audio_format, channels, samplerate,
        byte_rate, block_align, bits_per_sample,
    )

    # data chunk (8 + data)
    data_chunk = struct.pack("<4sI", b"data", data_size) + raw

    with open(path, "wb") as f:
        f.write(header + fmt_chunk + data_chunk)


def read_wav_info(path: str | Path) -> dict:
    """读取 WAV 文件元信息。

    Returns:
        {samplerate, channels, frames, duration, bits_per_sample}
    """
    path = Path(path)
    with open(path, "rb") as f:
        riff, size, wave = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"不是有效的 WAV 文件: {path}")

        samplerate = channels = bits_per_sample = None
        data_size = 0

        while f.tell() < size + 8:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                (audio_format, channels, samplerate,
                 byte_rate, block_align, bits_per_sample) = struct.unpack("<HHIIHH", f.read(16))
            elif chunk_id == b"data":
                data_size = chunk_size
                break
            else:
                f.seek(chunk_size, 1)

        if samplerate is None or data_size == 0:
            raise ValueError(f"无法解析 WAV 元信息: {path}")

        frames = data_size // (bits_per_sample // 8) // channels
        duration = frames / samplerate

        return {
            "samplerate": samplerate,
            "channels": channels,
            "frames": frames,
            "duration": duration,
            "bits_per_sample": bits_per_sample,
        }


def read_audio_info(path: str | Path, ffmpeg_path: str | None = None) -> dict:
    """读取任意音频格式的元信息。

    WAV 直接解析头（快）；其他格式用 ffmpeg 解析 stderr。

    Returns:
        {samplerate, channels, frames, duration}
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".wav":
        info = read_wav_info(path)
        return {k: info[k] for k in ("samplerate", "channels", "frames", "duration")}

    # 非 WAV：用 ffmpeg 解析
    if ffmpeg_path is None:
        from rvc.audio.loader import _ffmpeg
        ffmpeg_path = str(_ffmpeg())

    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    )
    stderr = result.stderr

    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    if not dur_match:
        raise ValueError(f"ffmpeg 无法解析时长: {path}")

    h, m, s = dur_match.groups()
    duration = int(h) * 3600 + int(m) * 60 + float(s)

    stream_match = re.search(
        r"Stream #\d+:\d+.*Audio:\s*\S+,\s*(\d+)\s*Hz,\s*(\w+)",
        stderr,
    )
    if stream_match:
        samplerate = int(stream_match.group(1))
        ch_str = stream_match.group(2).lower()
        ch_map = {"mono": 1, "stereo": 2, "2.1": 3, "quad": 4, "5.1": 6, "7.1": 8}
        channels = ch_map.get(ch_str, 2)
    else:
        samplerate = 0
        channels = 0

    frames = int(duration * samplerate) if samplerate else 0

    return {
        "samplerate": samplerate,
        "channels": channels,
        "frames": frames,
        "duration": duration,
    }
