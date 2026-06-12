import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from rvc.audio.loader import load_audio as _load_audio_lib

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FFMPEG = _PROJECT_ROOT / "assets" / "ffmpeg" / "ffmpeg.exe"
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
_MANIFEST_NAME = "manifest.json"
_RUNTIME_DIRS = [
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature768",
    "filelist.txt",
]
_CHECKPOINT_GLOBS = ["G_*.pth", "D_*.pth"]


class Slicer:
    def __init__(self, sr: int, threshold: float = -42.0, min_length: int = 1500, min_interval: int = 400, hop_size: int = 15, max_sil_kept: int = 500):
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.0)
        self.min_length = int(sr * min_length / 1000)
        self.min_interval = int(sr * min_interval / 1000)
        self.hop_size = int(sr * hop_size / 1000)
        self.max_sil_kept = int(sr * max_sil_kept / 1000)

    def slice(self, wav: np.ndarray):
        if wav.shape[0] <= self.min_length:
            return [wav]
        rms = librosa.feature.rms(y=wav, frame_length=self.hop_size * 2, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, value in enumerate(rms):
            pos = i * self.hop_size
            if value < self.threshold:
                if silence_start is None:
                    silence_start = pos
                continue
            if silence_start is None:
                continue
            silence_end = pos
            silence_len = silence_end - silence_start
            if silence_len >= self.min_interval and silence_end - clip_start >= self.min_length:
                cut = silence_start + min(silence_len // 2, self.max_sil_kept)
                sil_tags.append((clip_start, cut))
                clip_start = cut
            silence_start = None
        if clip_start < wav.shape[0]:
            sil_tags.append((clip_start, wav.shape[0]))
        return [wav[start:end] for start, end in sil_tags if end - start > self.hop_size]


class PreProcessor:
    def __init__(self, input_dir: str, exp_dir: str, sr: int, per: float = 3.7):
        self.input_dir = Path(input_dir)
        self.exp_dir = Path(exp_dir)
        self.sr = sr
        self.per = per
        self.gt_dir = self.exp_dir / "0_gt_wavs"
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        self.slicer = Slicer(sr)
        self.bh, self.ah = signal.butter(5, 48, btype="high", fs=sr)

    def run(self, progress_callback=None):
        self._prepare_exp_dir()
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.wav16k_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in self.input_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTS]
        for i, path in enumerate(files, 1):
            self._process_file(path, i)
            if progress_callback:
                progress_callback(i, len(files))
        return len(files)

    def _prepare_exp_dir(self):
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        current = build_exp_manifest(self.input_dir, self.sr, self.per)
        manifest_path = self.exp_dir / _MANIFEST_NAME
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_key = {k: previous.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
            current_key = {k: current.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
            if previous_key != current_key:
                clear_exp_runtime(self.exp_dir)
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    def _process_file(self, path: Path, file_index: int):
        wav, _ = _load_audio_lib(path, self.sr)
        if wav.size == 0:
            return
        wav = signal.lfilter(self.bh, self.ah, wav).astype(np.float32)
        pieces = self.slicer.slice(wav)
        chunk_len = int(self.per * self.sr)
        overlap = int(0.3 * self.sr)
        idx = 0
        for piece in pieces:
            start = 0
            while start < len(piece):
                chunk = piece[start : start + chunk_len]
                if len(chunk) < self.sr:
                    break
                chunk = normalize_audio(chunk)
                name = f"{file_index}_{idx}.wav"
                sf.write(self.gt_dir / name, chunk, self.sr, subtype="FLOAT")
                chunk16 = librosa.resample(chunk, orig_sr=self.sr, target_sr=16000) if self.sr != 16000 else chunk
                sf.write(self.wav16k_dir / name, chunk16, 16000, subtype="FLOAT")
                idx += 1
                if start + chunk_len >= len(piece):
                    break
                start += chunk_len - overlap


def normalize_audio(wav: np.ndarray):
    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 2.5:
        return np.zeros(0, dtype=np.float32)
    if peak > 0:
        wav = wav / peak * 0.9 * 0.75 + wav * 0.25
    return wav.astype(np.float32)


def load_audio(path: str | Path, sr: int):
    """兼容旧接口，委托给 rvc.audio_loader。"""
    return _load_audio_lib(path, sr)


def build_exp_manifest(input_dir: str | Path, sr: int, per: float):
    root = Path(input_dir).resolve()
    files = sorted(
        f"{str(p.relative_to(root)).replace('\\', '/')}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
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
    for pattern in _CHECKPOINT_GLOBS:
        for path in exp.glob(pattern):
            path.unlink()


def manifest_matches(exp_dir: str | Path, input_dir: str | Path, sr: int, per: float):
    manifest_path = Path(exp_dir) / _MANIFEST_NAME
    if not manifest_path.exists():
        return False
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_exp_manifest(input_dir, sr, per)
    saved_key = {k: saved.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
    current_key = {k: current.get(k) for k in ("input_dir", "input_hash", "sr", "per")}
    return saved_key == current_key


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
