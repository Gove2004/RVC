import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import signal

from rvc.core.constants import (
    HUBERT_SAMPLE_RATE,
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

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
_MANIFEST_NAME = "manifest.json"

# 除 `per` 外参与预处理指纹的键；必须与 PreProcessor.preprocess_params() 的键集合一致。
# 改其中任意一项都必须重新预处理，否则新参数会被静默忽略、沿用旧切片。
PREPROCESS_FINGERPRINT_KEYS = (
    "overlap",
    "highpass",
    "slice_threshold",
    "slice_min_length",
    "slice_min_interval",
    "slice_hop_size",
    "slice_max_sil_kept",
)

# 本次改动之前生成的 manifest 没有 "preprocess" 段。比对时按内置默认值代入，
# 这样「从没改过这些参数」的老实验不会被平白触发一次重新预处理
# （重新预处理会连同 checkpoint 一起清掉，代价很高）。
_LEGACY_PREPROCESS = {
    "overlap": SLICE_OVERLAP_DEFAULT,
    "highpass": PREPROCESS_HIGHPASS_DEFAULT,
    "slice_threshold": SLICE_THRESHOLD_DEFAULT,
    "slice_min_length": SLICE_MIN_LENGTH_DEFAULT,
    "slice_min_interval": SLICE_MIN_INTERVAL_DEFAULT,
    "slice_hop_size": SLICE_HOP_SIZE_DEFAULT,
    "slice_max_sil_kept": SLICE_MAX_SIL_KEPT_DEFAULT,
}
_RUNTIME_DIRS = [
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature768",
    "spec_cache",  # A6：STFT 缓存独立目录，素材变更时随切片产物一起清除
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
    """切片 + 重采样。

    切片/滤波参数全部可配（训练向导逐项提问），默认值取自 `rvc.core.constants`
    ——Slicer 的签名默认值引用同一批常量，故「不传即为历史行为」。
    """

    def __init__(self, input_dir: str, exp_dir: str, sr: int, per: float = SLICE_PERIOD_DEFAULT,
                 ffmpeg_exe: str | Path | None = None, overlap: float = SLICE_OVERLAP_DEFAULT,
                 highpass: float = PREPROCESS_HIGHPASS_DEFAULT,
                 slice_threshold: float = SLICE_THRESHOLD_DEFAULT,
                 slice_min_length: int = SLICE_MIN_LENGTH_DEFAULT,
                 slice_min_interval: int = SLICE_MIN_INTERVAL_DEFAULT,
                 slice_hop_size: int = SLICE_HOP_SIZE_DEFAULT,
                 slice_max_sil_kept: int = SLICE_MAX_SIL_KEPT_DEFAULT):
        self.input_dir = Path(input_dir)
        self.exp_dir = Path(exp_dir)
        self.sr = sr
        self.per = per
        # 重叠 >= 切片时长时切片推进为零 → _process_file 会死循环，宁可早报错
        if not 0 <= overlap < per:
            raise ValueError(f"切片重叠需要在 0 ~ {per}s 之间（收到 {overlap}）")
        self.overlap = overlap
        self.highpass = highpass
        self.ffmpeg_exe = ffmpeg_exe
        self.gt_dir = self.exp_dir / "0_gt_wavs"
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        self.slicer = Slicer(
            sr,
            threshold=slice_threshold,
            min_length=slice_min_length,
            min_interval=slice_min_interval,
            hop_size=slice_hop_size,
            max_sil_kept=slice_max_sil_kept,
        )
        # Slicer 内部把阈值转成了线性幅度，这里另存一份原始参数用于指纹比对
        self.slice_params = {
            "slice_threshold": slice_threshold,
            "slice_min_length": slice_min_length,
            "slice_min_interval": slice_min_interval,
            "slice_hop_size": slice_hop_size,
            "slice_max_sil_kept": slice_max_sil_kept,
        }
        self.bh, self.ah = signal.butter(5, highpass, btype="high", fs=sr)

    def preprocess_params(self) -> dict:
        """除 `per` 外的切片/滤波参数，键与 PREPROCESS_FINGERPRINT_KEYS 一致。"""
        return {"overlap": self.overlap, "highpass": self.highpass, **self.slice_params}

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
        current = build_exp_manifest(self.input_dir, self.sr, self.per, self.preprocess_params())
        manifest_path = self.exp_dir / _MANIFEST_NAME
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = ("input_dir", "input_hash", "sr", "per", "preprocess")
            previous_key = {k: previous.get(k, _LEGACY_PREPROCESS if k == "preprocess" else None) for k in keys}
            current_key = {k: current.get(k) for k in keys}
            if previous_key != current_key:
                clear_exp_runtime(self.exp_dir)
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    def _process_file(self, path: Path, file_index: int):
        wav, _ = _load_audio_lib(path, self.sr, ffmpeg_exe=self.ffmpeg_exe)
        if wav.size == 0:
            return
        wav = signal.lfilter(self.bh, self.ah, wav).astype(np.float32)
        pieces = self.slicer.slice(wav)
        chunk_len = int(self.per * self.sr)
        overlap = int(self.overlap * self.sr)
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
    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 2.5:
        return np.zeros(0, dtype=np.float32)
    if peak > 0:
        wav = wav / peak * 0.9 * 0.75 + wav * 0.25
    return wav.astype(np.float32)


def build_exp_manifest(input_dir: str | Path, sr: int, per: float, preprocess: dict | None = None):
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
        "preprocess": preprocess if preprocess is not None else dict(_LEGACY_PREPROCESS),
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


def manifest_diff_reason(exp_dir: str | Path, input_dir: str | Path, sr: int, per: float,
                        preprocess: dict | None = None) -> str:
    """返回预处理指纹与当前素材/参数不一致的原因；一致时返回空串。

    preprocess 为本次要用的切片/滤波参数（键见 PREPROCESS_FINGERPRINT_KEYS）；
    缺省时按内置默认值比对（老调用点/老实验的兼容口径）。
    """
    manifest_path = Path(exp_dir) / _MANIFEST_NAME
    if not manifest_path.exists():
        return "该实验目录还没有预处理产物（manifest.json 缺失），需要先执行预处理"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_exp_manifest(input_dir, sr, per, preprocess)
    if str(saved.get("input_dir")) != str(current.get("input_dir")):
        return f"输入目录不一致（预处理: {saved.get('input_dir')}，当前: {current.get('input_dir')}）"
    if saved.get("sr") != current.get("sr"):
        return f"采样率不一致（{saved.get('sr')} vs {current.get('sr')}）"
    if saved.get("per") != current.get("per"):
        return f"切片时长不一致（{saved.get('per')} vs {current.get('per')}）"
    saved_pre = saved.get("preprocess", _LEGACY_PREPROCESS)
    if saved_pre != current["preprocess"]:
        changed = [k for k in PREPROCESS_FINGERPRINT_KEYS
                   if (saved_pre or {}).get(k) != current["preprocess"][k]]
        detail = "、".join(changed) if changed else "切片/滤波参数"
        return f"预处理参数不一致（{detail}）"
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
