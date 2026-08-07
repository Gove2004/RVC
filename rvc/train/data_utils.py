import bisect
import logging
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from rvc.train.mel_processing import spectrogram_torch

logger = logging.getLogger(__name__)


class TextAudioLoaderMultiNSFsid(Dataset):
    def __init__(self, filelist_path: str, data_config: dict):
        self.data_config = data_config
        self.audiopaths_and_text = self._load_filelist(filelist_path)
        self.max_wav_value = data_config["max_wav_value"]
        self.sampling_rate = data_config["sampling_rate"]
        self.filter_length = data_config["filter_length"]
        self.hop_length = data_config["hop_length"]
        self.win_length = data_config["win_length"]
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
            length = np.load(feat_path, mmap_mode="r").shape[0] * 2
            if 100 <= length <= 900:
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
        return phone, pitch, pitchf, spec, wav, int(sid)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def _get_audio(self, filename: str):
        wav, sr = librosa.load(filename, sr=self.sampling_rate, mono=True)
        if sr != self.sampling_rate:
            raise ValueError(f"采样率不匹配: {sr} != {self.sampling_rate}")
        wav = torch.FloatTensor(wav).unsqueeze(0)
        spec_path = f"{filename}.sr{self.sampling_rate}.spec.pt"
        if Path(spec_path).exists():
            spec = torch.load(spec_path, map_location="cpu", weights_only=False)
        else:
            spec = spectrogram_torch(
                wav,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            ).squeeze(0)
            torch.save(spec, spec_path)
        return spec, wav.squeeze(0)


class TextAudioCollateMultiNSFsid:
    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[3].size(1), reverse=True)
        max_phone_len = max(x[0].size(0) for x in batch)
        max_spec_len = max(x[3].size(1) for x in batch)
        max_wav_len = max(x[4].size(0) for x in batch)
        spec_channels = batch[0][3].size(0)
        feat_dim = batch[0][0].size(1)
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

        for i, (phone_i, pitch_i, pitchf_i, spec_i, wav_i, sid_i) in enumerate(batch):
            phone_len = phone_i.size(0)
            spec_len = spec_i.size(1)
            wav_len = wav_i.size(0)
            phone[i, :phone_len] = phone_i
            phone_lengths[i] = phone_len
            pitch[i, :phone_len] = pitch_i
            pitchf[i, :phone_len] = pitchf_i
            spec[i, :, :spec_len] = spec_i
            spec_lengths[i] = spec_len
            wav[i, :wav_len] = wav_i
            wav_lengths[i] = wav_len
            sid[i] = sid_i
        return phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wav, wav_lengths, sid


class BucketSampler(Sampler):
    def __init__(self, dataset: TextAudioLoaderMultiNSFsid, batch_size: int, boundaries=None, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries or [100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.shuffle = shuffle
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
            logger.warning("BucketSampler: %d/%d 样本被跳过（长度超出范围 [%d, %d]）", dropped, total, self.boundaries[0], self.boundaries[-1])
        return [bucket for bucket in buckets if bucket]

    def _bisect(self, length: int):
        i = bisect.bisect_right(self.boundaries, length) - 1
        # 桶数量 = 边界区间数；不能用 self.buckets（初始化期间尚未赋值，且过滤空桶后长度会变）
        return i if 0 <= i < len(self.boundaries) - 1 else -1

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            ids = bucket.copy()
            if self.shuffle:
                random.shuffle(ids)
            for i in range(0, len(ids) - len(ids) % self.batch_size, self.batch_size):
                batches.append(ids[i : i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(len(bucket) // self.batch_size for bucket in self.buckets)
