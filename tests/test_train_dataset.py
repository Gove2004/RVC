"""训练数据装配契约 — 分桶采样不丢样本、样本内存缓存按预算淘汰。

这两条都属于「静默损失」类问题：语料量不是 batch_size 整数倍时每轮丢弃
桶尾样本（无提示、永不训练）；样本缓存若不计预算则会吃光内存。
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rvc.train.dataset import BucketSampler, TextAudioLoaderMultiNSFsid, _SampleMemoryCache


def _dataset(lengths):
    """BucketSampler 只依赖 dataset.lengths。"""
    return SimpleNamespace(lengths=list(lengths))


class TestBucketSamplerCoverage(unittest.TestCase):
    def test_every_sample_appears_exactly_once(self):
        lengths = [150] * 5 + [250] * 3 + [450] * 7 + [999] * 2   # 各不相同且非整除
        sampler = BucketSampler(_dataset(lengths), batch_size=3)

        seen = [idx for batch in sampler for idx in batch]
        self.assertEqual(sorted(seen), list(range(len(lengths))),
                         "分桶采样必须每轮覆盖全部样本，不许丢桶尾")

    def test_len_matches_iterated_batches(self):
        for lengths in ([150] * 5, [150] * 6, [200] * 1, [150, 250, 250, 450], []):
            sampler = BucketSampler(_dataset(lengths), batch_size=4)
            self.assertEqual(len(sampler), len(list(sampler)),
                             f"lengths={lengths} 时 __len__ 必须与 __iter__ 一致")

    def test_tail_batch_is_the_only_short_one(self):
        sampler = BucketSampler(_dataset([150] * 5), batch_size=3)
        sizes = sorted(len(batch) for batch in sampler)
        self.assertEqual(sizes, [2, 3], "只有尾批允许短一个 batch")

    def test_batch_size_one_keeps_all(self):
        sampler = BucketSampler(_dataset([150, 250, 450]), batch_size=1)
        self.assertEqual(len(sampler), 3)
        self.assertEqual(sorted(i for batch in sampler for i in batch), [0, 1, 2])


class TestSampleMemoryCache(unittest.TestCase):
    def test_lru_evicts_oldest_when_over_budget(self):
        cache = _SampleMemoryCache(budget_bytes=30)
        cache.put(0, "a", 10)
        cache.put(1, "b", 10)
        cache.put(2, "c", 10)
        cache.get(0)              # 刷新 0 的最近使用位置
        cache.put(3, "d", 10)     # 超预算 → 淘汰最久未用的 1

        self.assertEqual(cache.get(1), None)
        self.assertEqual(cache.get(0), "a")
        self.assertEqual(cache.get(3), "d")

    def test_oversized_entry_is_not_cached(self):
        cache = _SampleMemoryCache(budget_bytes=10)
        cache.put(0, "huge", 11)
        self.assertIsNone(cache.get(0))
        self.assertEqual(len(cache), 0)

    def test_zero_budget_disables_cache(self):
        cache = _SampleMemoryCache(budget_bytes=0)
        cache.put(0, "a", 1)
        self.assertIsNone(cache.get(0))

    def test_replacing_key_does_not_double_count(self):
        cache = _SampleMemoryCache(budget_bytes=20)
        cache.put(0, "a", 15)
        cache.put(0, "b", 15)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get(0), "b")


class TestDatasetCacheWiring(unittest.TestCase):
    """缓存必须在 __getitem__ 里生效，且命中时不碰磁盘。"""

    def _loader(self):
        loader = TextAudioLoaderMultiNSFsid.__new__(TextAudioLoaderMultiNSFsid)
        loader._mem_cache = _SampleMemoryCache(budget_bytes=1024 * 1024)
        return loader

    def test_second_access_hits_memory(self):
        from rvc.train.dataset import TrainSample

        loader = self._loader()
        sample = TrainSample(torch.zeros(4), torch.zeros(4, dtype=torch.long),
                             torch.zeros(3), torch.zeros(2, 4), torch.zeros(8), 0)
        with mock.patch.object(loader, "_load_sample", return_value=sample) as load:
            first = loader[3]
            second = loader[3]

        self.assertIs(first, second)
        self.assertEqual(load.call_count, 1, "第二个 epoch 不该重复读盘")

    def test_zero_budget_falls_back_to_disk_every_time(self):
        from rvc.train.dataset import TrainSample

        loader = self._loader()
        loader._mem_cache = _SampleMemoryCache(budget_bytes=0)
        sample = TrainSample(torch.zeros(4), torch.zeros(4, dtype=torch.long),
                             torch.zeros(3), torch.zeros(2, 4), torch.zeros(8), 0)
        with mock.patch.object(loader, "_load_sample", return_value=sample) as load:
            loader[0]
            loader[0]
        self.assertEqual(load.call_count, 2)


class TestTrainingDataLoaderOptions(unittest.TestCase):
    def test_defaults_use_bounded_prefetch_and_persistent_workers(self):
        from rvc.train.trainer import training_dataloader_options

        with mock.patch.dict("os.environ", {}, clear=True):
            options = training_dataloader_options("cuda:0")
        self.assertEqual(options["num_workers"], 2)
        self.assertEqual(options["prefetch_factor"], 1)
        self.assertTrue(options["persistent_workers"])
        self.assertEqual(options["mem_cache_bytes"], 128 * 1024 * 1024)

    def test_cleanup_shuts_down_persistent_workers(self):
        from rvc.train.trainer import Trainer

        iterator = mock.Mock()
        loader = mock.Mock(_iterator=iterator)
        trainer = Trainer.__new__(Trainer)
        trainer.loader = loader
        with mock.patch("torch.cuda.empty_cache"):
            trainer.cleanup()
        iterator._shutdown_workers.assert_called_once_with()
        self.assertIsNone(trainer.loader)

    def test_zero_workers_disable_worker_only_options(self):
        from rvc.train.trainer import training_dataloader_options

        with mock.patch.dict("os.environ", {
            "RVC_TRAIN_WORKERS": "0",
            "RVC_TRAIN_CACHE_MB": "0",
        }, clear=True):
            options = training_dataloader_options("cpu")
        self.assertEqual(options, {
            "num_workers": 0,
            "pin_memory": False,
            "mem_cache_bytes": 0,
        })

    def test_invalid_environment_values_fall_back(self):
        from rvc.train.trainer import training_dataloader_options

        with mock.patch.dict("os.environ", {
            "RVC_TRAIN_WORKERS": "bad",
            "RVC_TRAIN_PREFETCH": "bad",
            "RVC_TRAIN_CACHE_MB": "bad",
        }, clear=True):
            options = training_dataloader_options("cpu")
        self.assertEqual(options["num_workers"], 2)
        self.assertEqual(options["prefetch_factor"], 1)
        self.assertEqual(options["mem_cache_bytes"], 128 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
