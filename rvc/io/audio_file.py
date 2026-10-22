"""音频数据加载器 — 基于 ffmpeg 解码任意格式到 numpy 数组。

职责：把磁盘上的音频文件（mp3/flac/ogg/wav/m4a 等）解码并重采样为
float32 numpy 数组。这是"数据加载"，不是"文件读写"。

与 wav_file 的分工：
- audio_file.py: 解码任意格式 → numpy 数组（需要 ffmpeg）
- wav_file.py:   WAV 文件结构化读写（写 FLOAT32/PCM16，读元信息）

ffmpeg 定位单源在本文件 locate_ffmpeg（RVC_FFMPEG → 系统 PATH → assets/ffmpeg）：
训练 CLI 的 envcheck 探测与运行时默认路径共用同一查找链，避免两处各写一套。
"""
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from rvc.core.errors import AudioLoadError
from rvc.runtime.paths import FFMPEG_EXE

logger = logging.getLogger(__name__)

# pythonw（GUI）下启动子进程会闪控制台窗口，仅 Windows 需要抑制
_SUBPROCESS_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW

_FFMPEG_HINT = (
    "安装方式：winget install ffmpeg，或把 ffmpeg.exe 放到 assets/ffmpeg/ 目录下。"
    if os.name == "nt"
    else "安装方式：apt install -y ffmpeg（或 conda install -c conda-forge ffmpeg）。"
)

# resolve_ffmpeg 的默认路径缓存：{RVC_FFMPEG 取值: 解析结果}。只缓存一条
# （环境变量变了就换键），避免「改了环境变量还拿到旧 exe」的陷阱。
_resolved_default: dict[str, Path] = {}


def locate_ffmpeg() -> tuple[str, str]:
    """按固定顺序定位 ffmpeg，返回 (路径, 来源)；未找到返回 ("", "")。

    查找链（单源，训练 CLI envcheck 与运行时默认路径共用）：
    环境变量 RVC_FFMPEG → 系统 PATH → 项目内 assets/ffmpeg。
    项目内只随包提供 Windows 版 ffmpeg.exe，故第三级有意限定 Windows
    （非 Windows 平台交给系统 PATH / RVC_FFMPEG 处理）。
    """
    env_ffmpeg = os.environ.get("RVC_FFMPEG", "").strip()
    if env_ffmpeg and Path(env_ffmpeg).exists():
        return env_ffmpeg, "环境变量 RVC_FFMPEG"
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg, "系统 PATH"
    if os.name == "nt" and FFMPEG_EXE.exists():
        return str(FFMPEG_EXE), "项目内 assets/ffmpeg/ffmpeg.exe"
    return "", ""


def resolve_ffmpeg(ffmpeg_exe: str | Path | None = None) -> Path:
    """解析 ffmpeg 可执行路径。

    异常契约：
    - FileNotFoundError: 路径不存在/未找到。错误类型刻意与解码失败区分——
      "环境没装好"（调用方/用户可自行修复）和"文件坏"（数据问题）是两类故障。

    默认（ffmpeg_exe=None）走 locate_ffmpeg 查找链；显式传 ffmpeg_exe 是
    唯一的路径覆盖方式——不存在进程内改写模块全局的通道。

    默认路径结果按 RVC_FFMPEG 取值缓存：locate_ffmpeg 内部的 shutil.which
    要扫一遍 PATH，而 load_audio 在训练预处理阶段是「每个文件一次」——
    素材上千条时这是白跑的目录遍历。缓存键带上环境变量值，改了环境变量
    立即失效；命中时仍校验文件存在，被删了会自动重新定位。
    """
    if ffmpeg_exe is not None:
        exe = Path(ffmpeg_exe)
        if not exe.exists():
            raise FileNotFoundError(f"找不到 ffmpeg: {exe}\n{_FFMPEG_HINT}")
        return exe

    env_key = os.environ.get("RVC_FFMPEG", "").strip()
    cached = _resolved_default.get(env_key)
    if cached is not None and cached.exists():
        return cached

    found, _source = locate_ffmpeg()
    if not found:
        raise FileNotFoundError(f"找不到 ffmpeg（系统 PATH / assets/ffmpeg 都没有）\n{_FFMPEG_HINT}")
    resolved = Path(found)
    _resolved_default.clear()
    _resolved_default[env_key] = resolved
    return resolved


def load_audio(
    path: str | Path,
    target_sr: int,
    mono: bool = True,
    timeout: int = 300,
    ffmpeg_exe: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """用 ffmpeg 解码并重采样到 target_sr，输出 float32。

    mono=False 时返回 shape (2, N) 的立体声。
    timeout: ffmpeg 子进程超时秒数（默认 300，长音频按需调大）。

    异常契约：
    - FileNotFoundError: ffmpeg 可执行文件不存在（环境问题，见 resolve_ffmpeg）
    - AudioLoadError:    ffmpeg 返回非零退出码（文件损坏/格式不支持，数据问题）
    - subprocess.TimeoutExpired: 超过 timeout，原样抛出不吞
    """
    path = Path(path).resolve()
    cmd = [
        str(resolve_ffmpeg(ffmpeg_exe)), "-i", str(path), "-vn",
        "-acodec", "pcm_f32le", "-f", "f32le",
        "-ac", "1" if mono else "2", "-ar", str(target_sr), "-",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, timeout=timeout,
        creationflags=_SUBPROCESS_NO_WINDOW, stdin=subprocess.DEVNULL,
    )
    if proc.returncode:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioLoadError(f"ffmpeg 解码失败: {path}\n{err}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        raw = raw.reshape(-1, 2).T  # f32le 交织 LRLR… → (2, N)
    return raw.astype(np.float32), target_sr
