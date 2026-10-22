"""训练预处理链 — 边界与中断语义的回归测试。

锁定的都是「静默失败」类问题（不报错、但结果错），这类问题靠人工排查成本极高：
- normalize_audio 对削波素材返回空数组，调用方必须跳过而不是落空 wav
- BucketSampler 边界右端封闭：length == TRAIN_FRAME_MAX 的样本必须能入桶
- prune_orphan_checkpoints 顺带清掉 *.pairtmp 中断残留
- _prepare_exp_dir 在「有产物但无 manifest」时清空重建（旧切片会与新素材错位）
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvc.core.constants import TRAIN_FRAME_MAX, TRAIN_FRAME_MIN
from rvc.train.checkpoint import checkpoints_dir, prune_orphan_checkpoints
from rvc.train.dataset import BucketSampler
from rvc.train.preprocess import PreProcessor, normalize_audio


class _FakeDataset:
    """BucketSampler 只依赖 .lengths。"""

    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)


class TestNormalizeAudio(unittest.TestCase):
    def test_peak_over_threshold_returns_empty(self):
        """削波素材必须返回空数组（由调用方跳过），而不是被归一化后混入训练集。"""
        loud = np.full(1000, 3.0, dtype=np.float32)
        self.assertEqual(normalize_audio(loud).size, 0)

    def test_normal_peak_normalized(self):
        wav = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
        out = normalize_audio(wav)
        self.assertEqual(out.shape, wav.shape)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)


class TestBucketSamplerBoundaries(unittest.TestCase):
    def _sampler(self, lengths, batch_size=2):
        return BucketSampler(_FakeDataset(lengths), batch_size)

    def test_max_length_is_not_dropped(self):
        """length == TRAIN_FRAME_MAX 必须落进合法桶，不能算出桶下标 == 桶数被静默丢弃。"""
        sampler = self._sampler([TRAIN_FRAME_MAX])
        assigned = [i for bucket in sampler.buckets for i in bucket]
        self.assertEqual(assigned, [0], "等于上界的样本必须入桶")

    def test_range_edges_land_in_buckets(self):
        lengths = [TRAIN_FRAME_MIN, TRAIN_FRAME_MIN + 1, TRAIN_FRAME_MAX - 1, TRAIN_FRAME_MAX]
        sampler = self._sampler(lengths)
        assigned = sorted(i for bucket in sampler.buckets for i in bucket)
        self.assertEqual(assigned, [0, 1, 2, 3])

    def test_out_of_range_still_rejected(self):
        """低于下界的样本仍应被拒（_filter 本不该放它进来，防御性保留）。"""
        sampler = self._sampler([TRAIN_FRAME_MIN - 1])
        self.assertEqual(sampler.buckets, [])

    def test_boundaries_are_strictly_above_max(self):
        """最后一个边界必须严格大于上界，否则 bisect_right 会把上界样本挤出桶。"""
        sampler = self._sampler([TRAIN_FRAME_MAX])
        self.assertGreater(sampler.boundaries[-1], TRAIN_FRAME_MAX)


class TestPruneOrphanCheckpoints(unittest.TestCase):
    def test_removes_pairtmp_residue_and_keeps_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = checkpoints_dir(Path(d))
            ckpt.mkdir(parents=True)
            (ckpt / "G_1.pth").write_bytes(b"g")
            (ckpt / "D_1.pth").write_bytes(b"d")
            # 成对保存被中断的两种残骸
            (ckpt / "G_2.pth.pairtmp").write_bytes(b"g")
            (ckpt / "G_2.pth.pairtmp.tmp").write_bytes(b"g")

            removed = prune_orphan_checkpoints(d)

            left = sorted(p.name for p in ckpt.iterdir())
            self.assertEqual(left, ["D_1.pth", "G_1.pth"])
            self.assertEqual(removed, 2)

    def test_removes_single_sided_orphan(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = checkpoints_dir(Path(d))
            ckpt.mkdir(parents=True)
            (ckpt / "G_3.pth").write_bytes(b"g")

            self.assertEqual(prune_orphan_checkpoints(d), 1)
            self.assertEqual(list(ckpt.iterdir()), [])


class TestPrepareExpDirClearsWithoutManifest(unittest.TestCase):
    """无 manifest 但有产物 → 必须清空重建（否则旧切片与新素材错位混训）。"""

    def test_stale_artifacts_are_cleared(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d) / "exp"
            gt = exp / "0_gt_wavs"
            gt.mkdir(parents=True)
            stale = gt / "0_0.wav"
            stale.write_bytes(b"stale")

            processor = PreProcessor(str(Path(d)), str(exp), 40000)
            processor._prepare_exp_dir()

            self.assertFalse(stale.exists(), "无 manifest 时旧切片必须被清掉")

    def test_manifest_kept_when_fingerprint_matches(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "in"
            src.mkdir()
            exp = Path(d) / "exp"
            processor = PreProcessor(str(src), str(exp), 40000)
            processor._prepare_exp_dir()  # 写 pending
            (exp / "manifest.json.pending").replace(exp / "manifest.json")
            gt = exp / "0_gt_wavs"
            gt.mkdir(parents=True, exist_ok=True)
            keep = gt / "0_0.wav"
            keep.write_bytes(b"keep")

            PreProcessor(str(src), str(exp), 40000)._prepare_exp_dir()
            self.assertTrue(keep.exists(), "指纹一致时不得清空产物")


if __name__ == "__main__":
    unittest.main()
