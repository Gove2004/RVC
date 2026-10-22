"""训练数据集 — 文件清单加载、特征/频谱读取、分桶采样与 batch 拼接。

TextAudioLoaderMultiNSFsid 按帧长过滤样本，BucketSampler 按长度分桶减少
batch 内 padding，TextAudioCollateMultiNSFsid 负责补齐并打包。
"""
import bisect
import logging
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from rvc.core.constants import TRAIN_FRAME_MAX, TRAIN_FRAME_MIN
from rvc.io.audio_file import load_audio
from torch.utils.data import Dataset, Sampler

from rvc.train.atomic_save import save_npy_atomic, save_torch_atomic
from rvc.train.mel_processing import spectrogram_torch

logger = logging.getLogger(__name__)


class TrainSample(NamedTuple):
    """单条训练样本（`__getitem__` 的输出契约）。字段顺序即历史位置顺序。"""

    phone: torch.Tensor
    pitch: torch.Tensor
    pitchf: torch.Tensor
    spec: torch.Tensor
    wav: torch.Tensor
    sid: int


class TrainBatch(NamedTuple):
    """拼接后的 batch（collate 输出、trainer 消费的契约）。

    必须保持 NamedTuple：DataLoader(pin_memory=True) 的默认钉页实现只对
    namedtuple 逐元素重建（torch/utils/data/_utils/pin_memory.py:87-90），
    普通容器会被原样返回、内部张量不钉页，改变 non_blocking 拷贝语义。
    """

    phone: torch.Tensor
    phone_lengths: torch.Tensor
    pitch: torch.Tensor
    pitchf: torch.Tensor
    spec: torch.Tensor
    spec_lengths: torch.Tensor
    wav: torch.Tensor
    wav_lengths: torch.Tensor
    sid: torch.Tensor


class TextAudioLoaderMultiNSFsid(Dataset):
    def __init__(self, filelist_path: str, mel_config: dict, spec_cache_dir: str | Path,
                 ffmpeg_exe: str | None = None):
        """数据集加载器。

        Args:
            spec_cache_dir: STFT 频谱缓存目录。A6：缓存不再写进切片产物
                目录（0_gt_wavs/），统一放 {exp}/spec_cache/——切片目录保持
                "纯产物"，缓存失效由 mtime 判断 + 素材变更时整目录清除。
            ffmpeg_exe: ffmpeg 路径（D4 显式传参）。None = 用 runtime 默认。
        """
        self.mel_config = mel_config
        self.spec_cache_dir = Path(spec_cache_dir)
        self.ffmpeg_exe = ffmpeg_exe
        self.audiopaths_and_text = self._load_filelist(filelist_path)
        self.sampling_rate = mel_config["sampling_rate"]
        self.filter_length = mel_config["filter_length"]
        self.hop_length = mel_config["hop_length"]
        self.win_length = mel_config["win_length"]
        self.lengths = self._filter()

    @staticmethod
    def _load_filelist(filelist_path: str):
        lines = Path(filelist_path).read_text(encoding="utf-8").splitlines()
        return [line.split("|") for line in lines if line.strip()]

    def _filter(self):
        filtered = []
        lengths = []
        for item in self.audiopaths_and_text:
            if len(item) != 5:
                continue
            wav_path, feat_path, pitch_path, pitchf_path, sid = item
            if not all(Path(p).exists() for p in [wav_path, feat_path, pitch_path, pitchf_path]):
                continue
            # 长度 = 特征帧数 ×2（phone 会 repeat_interleave 两倍），
            # 有效范围单源 rvc.core/constants.py（与 BucketSampler 边界表一致）
            length = np.load(feat_path, mmap_mode="r").shape[0] * 2
            if TRAIN_FRAME_MIN <= length <= TRAIN_FRAME_MAX:
                filtered.append(item)
                lengths.append(length)
        self.audiopaths_and_text = filtered
        return lengths

    def __getitem__(self, index):
        wav_path, feat_path, pitch_path, pitchf_path, sid = self.audiopaths_and_text[index]
        phone = torch.from_numpy(np.load(feat_path).astype(np.float32))
        phone = phone.repeat_interleave(2, dim=0)
        pitch = torch.from_numpy(np.load(pitch_path).astype(np.int64))
        pitchf = torch.from_numpy(np.load(pitchf_path).astype(np.float32))
        spec, wav = self._get_audio(wav_path)

        min_len = min(phone.shape[0], pitch.shape[0], pitchf.shape[0], spec.shape[1])
        max_len = max(phone.shape[0], pitch.shape[0], pitchf.shape[0], spec.shape[1])
        # 帧数取整差异（HuBERT/RMVPE/STFT 各 ±1~2 帧）在 RVC 训练中属正常，截断量 <1% 无影响；
        # 仅明显不对齐（>4 帧）才告警，避免正常训练刷屏
        if max_len - min_len > 4:
            logger.warning("特征长度不对齐 (phone=%d, pitch=%d, pitchf=%d, spec=%d)，截断至 %d", phone.shape[0], pitch.shape[0], pitchf.shape[0], spec.shape[1], min_len)
        phone = phone[:min_len]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]
        spec = spec[:, :min_len]
        return TrainSample(phone, pitch, pitchf, spec, wav, int(sid))

    def __len__(self):
        return len(self.audiopaths_and_text)

    def _get_audio(self, filename: str):
        """解码 wav（落盘缓存）→ 计算/读取 STFT 频谱（落盘缓存）。

        两级缓存都放 spec_cache_dir，mtime 判断失效（素材变更时整目录清除）：
        - wav 缓存：消除每个 epoch 重复 spawn ffmpeg 解码同一文件
          （训练循环里最大的重复 I/O，约 50ms/次 × 样本数 × epoch 数）
        - spec 缓存：消除每个 epoch 重复 STFT
        """
        wav_mtime = Path(filename).stat().st_mtime
        wav = self._cached_wav(filename, wav_mtime)
        spec = self._cached_spec(filename, wav, wav_mtime)
        return spec, wav.squeeze(0)

    def _cached_wav(self, filename: str, mtime: float) -> torch.Tensor:
        """解码并重采样 wav，结果缓存为 npy（key 含采样率，mtime 判失效）。

        load_audio 的契约保证返回采样率恒等于 target_sr（ffmpeg -ar 强制重采样），
        无需二次校验。
        """
        cache_path = self.spec_cache_dir / f"{Path(filename).stem}.sr{self.sampling_rate}.wav.npy"
        if cache_path.exists() and cache_path.stat().st_mtime >= mtime:
            # 安全前提：wav 缓存由本进程 save_npy_atomic 就近写出，来源可信
            return torch.from_numpy(np.load(cache_path)).unsqueeze(0)
        wav, _sr = load_audio(filename, target_sr=self.sampling_rate, mono=True, ffmpeg_exe=self.ffmpeg_exe)
        self.spec_cache_dir.mkdir(parents=True, exist_ok=True)
        # 原子写：中断不留半写缓存（tmp 名固定，依赖「同一路径不并发写」前提，
        # 见 atomic_save._atomic_write 的并发说明）
        save_npy_atomic(cache_path, wav.astype(np.float32))
        return torch.from_numpy(wav).unsqueeze(0)

    def _cached_spec(self, filename: str, wav: torch.Tensor, mtime: float) -> torch.Tensor:
        """计算 STFT 频谱，缓存 key 包含全部 STFT 参数（重新预处理自动失效）。"""
        spec_name = (
            f"{Path(filename).stem}.sr{self.sampling_rate}"
            f".n{self.filter_length}.h{self.hop_length}.w{self.win_length}.spec.pt"
        )
        spec_path = self.spec_cache_dir / spec_name
        cache_valid = (
            spec_path.exists()
            and spec_path.stat().st_mtime >= mtime
        )
        if cache_valid:
            # 安全前提：spec 缓存由本进程 save_torch_atomic 就近写出，来源可信，故 weights_only=False。
            return torch.load(spec_path, map_location="cpu", weights_only=False)
        spec = spectrogram_torch(
            wav,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
        ).squeeze(0)
        self.spec_cache_dir.mkdir(parents=True, exist_ok=True)
        # 原子写：与 wav 缓存同防护等级——中途强杀不留截断 .spec.pt
        # （非原子写会让损坏文件通过 mtime 判定静默复用，torch.load 直接崩）
        save_torch_atomic(spec_path, spec)
        return spec


class TextAudioCollateMultiNSFsid:
    def __call__(self, batch: list[TrainSample]) -> TrainBatch:
        batch = sorted(batch, key=lambda x: x.spec.size(1), reverse=True)
        max_phone_len = max(x.phone.size(0) for x in batch)
        max_spec_len = max(x.spec.size(1) for x in batch)
        max_wav_len = max(x.wav.size(0) for x in batch)
        spec_channels = batch[0].spec.size(0)
        feat_dim = batch[0].phone.size(1)
        b = len(batch)

        phone = torch.zeros(b, max_phone_len, feat_dim)
        phone_lengths = torch.LongTensor(b)
        pitch = torch.zeros(b, max_phone_len, dtype=torch.long)
        pitchf = torch.zeros(b, max_phone_len)
        spec = torch.zeros(b, spec_channels, max_spec_len)
        spec_lengths = torch.LongTensor(b)
        wav = torch.zeros(b, max_wav_len)
        wav_lengths = torch.LongTensor(b)
        sid = torch.LongTensor(b)

        for i, sample in enumerate(batch):
            phone_len = sample.phone.size(0)
            spec_len = sample.spec.size(1)
            wav_len = sample.wav.size(0)
            phone[i, :phone_len] = sample.phone
            phone_lengths[i] = phone_len
            pitch[i, :phone_len] = sample.pitch
            pitchf[i, :phone_len] = sample.pitchf
            spec[i, :, :spec_len] = sample.spec
            spec_lengths[i] = spec_len
            wav[i, :wav_len] = sample.wav
            wav_lengths[i] = wav_len
            sid[i] = sample.sid
        return TrainBatch(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wav, wav_lengths, sid)


class BucketSampler(Sampler):
    _BUCKET_STEP = 100

    def __init__(self, dataset: TextAudioLoaderMultiNSFsid, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        # 边界表从有效长度范围单源推导，**右端封闭**：bisect_right 返回的是
        # 「第一个 > length 的边界下标」，要让 TRAIN_FRAME_MAX 恰好落进最后一个桶，
        # 必须有一个严格大于它的边界。所以上界取 MAX + STEP 而非常见的 MAX + 1 段。
        # （历史 bug：range(MIN, MAX+1, STEP) 的末元素 = MAX，length == MAX 的样本
        # 进得了数据集、却算出桶下标 == 桶数而被静默丢弃，警告文案还自相矛盾。）
        self.boundaries = list(range(
            TRAIN_FRAME_MIN, TRAIN_FRAME_MAX + self._BUCKET_STEP + 1, self._BUCKET_STEP,
        ))
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        total = 0
        assigned = 0
        for idx, length in enumerate(self.dataset.lengths):
            total += 1
            bucket_idx = self._bisect(length)
            if bucket_idx != -1:
                assigned += 1
                buckets[bucket_idx].append(idx)
        if dropped := total - assigned:
            # boundaries[-1] 是开区间右端（不含），有效上界是它减一
            logger.warning(
                "BucketSampler：%d/%d 样本被跳过（长度超出范围 [%d, %d]）",
                dropped, total, self.boundaries[0], self.boundaries[-1] - 1,
            )
        return [bucket for bucket in buckets if bucket]

    def _bisect(self, length: int):
        i = bisect.bisect_right(self.boundaries, length) - 1
        # 桶数量 = 边界区间数；不能用 self.buckets（初始化期间尚未赋值，且过滤空桶后长度会变）
        return i if 0 <= i < len(self.boundaries) - 1 else -1

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            ids = bucket.copy()
            random.shuffle(ids)
            for i in range(0, len(ids) - len(ids) % self.batch_size, self.batch_size):
                batches.append(ids[i : i + self.batch_size])
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(len(bucket) // self.batch_size for bucket in self.buckets)
