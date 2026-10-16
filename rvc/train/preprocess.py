"""预处理 — 音频切片、滤波重采样与实验目录指纹核对。

切片/滤波/运行时默认值单源在 rvc/core/constants.py，本文件不写字面量。
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import signal

from rvc.core.constants import (
    AUDIO_EXTS,
    BAD_SAMPLE_PEAK,
    HUBERT_SAMPLE_RATE,
    NORMALIZE_DYNAMIC_MIX,
    NORMALIZE_PEAK_DEFAULT,
    PREPROCESS_HIGHPASS_DEFAULT,
    SLICE_HOP_SIZE_DEFAULT,
    SLICE_MAX_SIL_KEPT_DEFAULT,
    SLICE_MIN_INTERVAL_DEFAULT,
    SLICE_MIN_LENGTH_DEFAULT,
    SLICE_OVERLAP_DEFAULT,
    SLICE_PERIOD_DEFAULT,
    SLICE_THRESHOLD_DEFAULT,
)
from rvc.io.audio_file import load_audio as _load_audio_lib
from rvc.io.wav_file import write_wav
from rvc.train.checkpoint import CHECKPOINT_DIR_NAME

_MANIFEST_NAME = "manifest.json"
_MANIFEST_PENDING = "manifest.json.pending"
_RUNTIME_DIRS = [
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature768",
    "spec_cache",  # STFT 缓存独立目录，素材变更时随切片产物一起清除
    "filelist.txt",
]
_CHECKPOINT_GLOBS = ["G_*.pth", "D_*.pth"]



def _rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """计算每帧 RMS（兼容 librosa.feature.rms，center=True，reflect padding）。

    用于音频切片的静音检测，微小差异不影响最终结果。
    """
    pad = int(frame_length // 2)
    y_padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y_padded) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(n_frames, frame_length),
        strides=(y_padded.strides[0] * hop_length, y_padded.strides[0]),
    )
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)


def _load_manifest(manifest_path: Path) -> dict | None:
    """读 manifest.json；缺失或损坏（坏 JSON）返回 None——按「无 manifest」处理，
    与全项目配置类文件的「备份/回退」口径一致，不让一次写坏直接 crash。"""
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class Slicer:
    def __init__(self, sr: int, threshold: float = SLICE_THRESHOLD_DEFAULT,
                 min_length: int = SLICE_MIN_LENGTH_DEFAULT,
                 min_interval: int = SLICE_MIN_INTERVAL_DEFAULT,
                 hop_size: int = SLICE_HOP_SIZE_DEFAULT,
                 max_sil_kept: int = SLICE_MAX_SIL_KEPT_DEFAULT):
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.0)
        self.min_length = int(sr * min_length / 1000)
        self.min_interval = int(sr * min_interval / 1000)
        self.hop_size = int(sr * hop_size / 1000)
        self.max_sil_kept = int(sr * max_sil_kept / 1000)

    def slice(self, wav: np.ndarray):
        if wav.shape[0] <= self.min_length:
            return [wav]
        rms = _rms(wav, frame_length=self.hop_size * 2, hop_length=self.hop_size)
        below = rms < self.threshold
        length = len(below)

        sil_tags = []
        silence_start = None
        clip_start = 0
        i = 0
        while i < length:
            if below[i]:
                if silence_start is None:
                    silence_start = int(np.clip(i * self.hop_size, 0, length * self.hop_size))
                i += 1
                continue
            if silence_start is None:
                i += 1
                continue
            silence_end = min(int(np.clip((i + 1) * self.hop_size, 0, length * self.hop_size)), len(wav))
            silence_len = silence_end - silence_start
            if silence_len >= self.min_interval and silence_end - clip_start >= self.min_length:
                cut = silence_start + min(silence_len // 2, self.max_sil_kept)
                sil_tags.append((clip_start, cut))
                clip_start = cut
            silence_start = None
            i += 1
        if clip_start < len(wav):
            sil_tags.append((clip_start, len(wav)))
        return [wav[start:end] for start, end in sil_tags if end - start > self.hop_size]


class PreProcessor:
    """切片 + 重采样预处理器。

    参数面只剩 per（切片时长秒数）；overlap 等固定回常量 SLICE_OVERLAP_DEFAULT。
    注意 overlap >= per 时滑动窗口推进为零、会死循环——重开该参数必须在
    构造时 raise 拒绝，不能只告警。
    """

    def __init__(self, input_dir: str, exp_dir: str, sr: int, per: float = SLICE_PERIOD_DEFAULT,
                 ffmpeg_exe: str | Path | None = None):
        self.input_dir = Path(input_dir)
        self.exp_dir = Path(exp_dir)
        self.sr = sr
        self.per = per
        self.ffmpeg_exe = ffmpeg_exe
        self.gt_dir = self.exp_dir / "0_gt_wavs"
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        self.slicer = Slicer(sr)
        self.bh, self.ah = signal.butter(5, PREPROCESS_HIGHPASS_DEFAULT, btype="high", fs=sr)

    def run(self, progress_callback=None, stop_check=None):
        self._prepare_exp_dir()
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.wav16k_dir.mkdir(parents=True, exist_ok=True)
        # sorted 保证文件序号（切片文件名 {index}_{idx}.wav）确定性——中断后重跑
        # 才能得到同样的命名，不会留下新旧混排的半套产物
        files = sorted(p for p in self.input_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        for i, path in enumerate(files, 1):
            if stop_check is not None and stop_check():
                break
            self._process_file(path, i)
            if progress_callback:
                progress_callback(i, len(files))
        # 全部切片完成（未被中止）才把 pending 转正。manifest.json 的语义 =
        # 「预处理完整完成」：中途崩溃/强退/用户中止时 manifest.json 保持旧值
        # （或缺失），下次指纹核对必然判定需要重跑——不会静默用残缺的切片产物训练。
        if stop_check is None or not stop_check():
            pending = self.exp_dir / _MANIFEST_PENDING
            if pending.exists():
                os.replace(pending, self.exp_dir / _MANIFEST_NAME)
        return len(files)

    def _prepare_exp_dir(self):
        # 预处理指纹 = (input_dir, input_hash, sr, per)：与上次不一致则清空
        # 切片产物与 checkpoint（旧 checkpoint 对应旧素材，留着会从错误数据续训）
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        current = build_exp_manifest(self.input_dir, self.sr, self.per)
        manifest_path = self.exp_dir / _MANIFEST_NAME
        previous = _load_manifest(manifest_path)
        if previous is not None:
            previous_key = {k: previous.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
            current_key = {k: current.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
            if previous_key != current_key:
                clear_exp_runtime(self.exp_dir)
        # 先写 pending，run() 完成后才转正（见 run 里的说明）
        (self.exp_dir / _MANIFEST_PENDING).write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _process_file(self, path: Path, file_index: int):
        wav, _ = _load_audio_lib(path, self.sr, ffmpeg_exe=self.ffmpeg_exe)
        if wav.size == 0:
            return
        wav = signal.lfilter(self.bh, self.ah, wav).astype(np.float32)
        pieces = self.slicer.slice(wav)
        chunk_len = int(self.per * self.sr)
        overlap = int(SLICE_OVERLAP_DEFAULT * self.sr)
        idx = 0
        for piece in pieces:
            start = 0
            while start < len(piece):
                chunk = piece[start : start + chunk_len]
                if len(chunk) < self.sr:
                    break
                chunk = normalize_audio(chunk)
                name = f"{file_index}_{idx}.wav"
                write_wav(self.gt_dir / name, chunk, self.sr, subtype="FLOAT")
                chunk16 = signal.resample(chunk, int(len(chunk) * HUBERT_SAMPLE_RATE / self.sr)) if self.sr != HUBERT_SAMPLE_RATE else chunk
                write_wav(self.wav16k_dir / name, chunk16, HUBERT_SAMPLE_RATE, subtype="FLOAT")
                idx += 1
                if start + chunk_len >= len(piece):
                    break
                start += chunk_len - overlap


def normalize_audio(wav: np.ndarray):
    """峰值归一化并混回部分原始动态；峰值过高视为坏样本直接丢弃。

    参数单源 rvc/core/constants.py（NORMALIZE_PEAK_DEFAULT / NORMALIZE_DYNAMIC_MIX /
    BAD_SAMPLE_PEAK），影响训练素材的字面量不散落。
    """
    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > BAD_SAMPLE_PEAK:
        return np.zeros(0, dtype=np.float32)
    if peak > 0:
        wav = (wav / peak * NORMALIZE_PEAK_DEFAULT * NORMALIZE_DYNAMIC_MIX
               + wav * (1.0 - NORMALIZE_DYNAMIC_MIX))
    return wav.astype(np.float32)


def build_exp_manifest(input_dir: str | Path, sr: int, per: float):
    root = Path(input_dir).resolve()
    files = sorted(
        f"{str(p.relative_to(root)).replace('\\', '/')}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    digest = hashlib.sha1("\n".join(files).encode("utf-8")).hexdigest()
    return {
        "input_dir": str(root),
        "input_hash": digest,
        "sr": sr,
        "per": per,
    }


def clear_exp_runtime(exp_dir: str | Path):
    exp = Path(exp_dir)
    for name in _RUNTIME_DIRS:
        path = exp / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    # 根目录与 4_checkpoints/ 都要清：素材变了旧 checkpoint 已经失效，
    # 留下会让训练从错误的数据上续训
    targets = [exp, exp / CHECKPOINT_DIR_NAME]
    for pattern in _CHECKPOINT_GLOBS:
        for directory in targets:
            if not directory.is_dir():
                continue
            for path in directory.glob(pattern):
                path.unlink()


def manifest_diff_reason(exp_dir: str | Path, input_dir: str | Path, sr: int, per: float) -> str:
    """返回预处理指纹与当前素材不一致的原因；一致时返回空串。"""
    saved = _load_manifest(Path(exp_dir) / _MANIFEST_NAME)
    if saved is None:
        return "该实验目录还没有预处理产物（manifest.json 缺失或损坏），需要先执行预处理"
    current = build_exp_manifest(input_dir, sr, per)
    if str(saved.get("input_dir")) != str(current.get("input_dir")):
        return f"输入目录不一致（预处理: {saved.get('input_dir')}，当前: {current.get('input_dir')}）"
    if saved.get("sr") != current.get("sr"):
        return f"采样率不一致（{saved.get('sr')} vs {current.get('sr')}）"
    if saved.get("per") != current.get("per"):
        return f"切片时长不一致（{saved.get('per')} vs {current.get('per')}）"
    if saved.get("input_hash") != current.get("input_hash"):
        return "素材文件有变化（新增/删除/大小或修改时间变动），旧特征已过期"
    return ""


def generate_filelist(exp_dir: str):
    exp = Path(exp_dir)
    gt_dir = exp / "0_gt_wavs"
    feat_dir = exp / "3_feature768"
    f0_dir = exp / "2a_f0"
    f0nsf_dir = exp / "2b-f0nsf"
    lines = []
    for wav in sorted(gt_dir.glob("*.wav")):
        stem = wav.stem
        feat = feat_dir / f"{stem}.npy"
        f0 = f0_dir / f"{stem}.npy"
        f0nsf = f0nsf_dir / f"{stem}.npy"
        if feat.exists() and f0.exists() and f0nsf.exists():
            lines.append(f"{wav}|{feat}|{f0}|{f0nsf}|0")
    filelist = exp / "filelist.txt"
    filelist.write_text("\n".join(lines), encoding="utf-8")
    return filelist, len(lines)
