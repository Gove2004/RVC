import os
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FFMPEG = _PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


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
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.wav16k_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in self.input_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTS]
        for i, path in enumerate(files, 1):
            self._process_file(path, i)
            if progress_callback:
                progress_callback(i, len(files))
        return len(files)

    def _process_file(self, path: Path, file_index: int):
        wav, _ = load_audio(path, self.sr)
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
    path = Path(path)
    try:
        return librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        if not _FFMPEG.exists():
            raise FileNotFoundError(f"找不到 ffmpeg: {_FFMPEG}")
        info = subprocess.run([str(_FFMPEG), "-i", str(path)], capture_output=True, text=True)
        source_sr = sr
        for line in info.stderr.split("\n"):
            if "Hz" in line and "Audio" in line:
                m = re.search(r"(\d+) Hz", line)
                if m:
                    source_sr = int(m.group(1))
                    break
        cmd = [str(_FFMPEG), "-i", str(path), "-vn", "-acodec", "pcm_f32le", "-f", "f32le", "-ac", "1", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode:
            raise RuntimeError("ffmpeg 解码失败")
        wav = np.frombuffer(proc.stdout, dtype=np.float32)
        if source_sr != sr:
            wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sr)
        return wav.astype(np.float32), sr


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
