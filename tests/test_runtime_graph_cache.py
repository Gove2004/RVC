"""CUDA Graph 缓存层 — CPU 侧逻辑测试（捕获/回放需真 GPU，由实机覆盖）。

锁定：
- 同一签名命中缓存、不重复捕获
- LRU 按「最近使用」淘汰（不是按插入序）
- 非 CUDA 设备直接走 eager，不挂缓存对象
"""
import threading
import unittest
from unittest import mock

import torch

from rvc.runtime.graph import _GraphCache, run_cuda_graph


class _FakeEntry:
    """_CapturedCall 替身：不做真捕获，只记录回放。"""

    def __init__(self, function, inputs):
        self.lock = threading.RLock()
        self.function = function
        self.replays = 0

    def replay(self, inputs):
        self.replays += 1
        return self.function(*inputs)


def _sig_keys(cache):
    """缓存键的第一个元素（namespace），按 LRU 顺序（旧→新）。"""
    return [key[0] for key in cache.entries]


class TestGraphCache(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("rvc.runtime.graph._CapturedCall", _FakeEntry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cache = _GraphCache()
        self.cache.max_entries = 2
        self.tensor = torch.zeros(4, dtype=torch.float32)

    def _run(self, namespace):
        return self.cache.run((namespace,), lambda t: t + 1, (self.tensor,))

    def test_same_signature_reuses_entry(self):
        self._run("a")
        entry = next(iter(self.cache.entries.values()))
        self._run("a")
        self.assertEqual(entry.replays, 2, "同签名应回放而不是重新捕获")
        self.assertEqual(len(self.cache.entries), 1)

    def test_lru_evicts_least_recently_used(self):
        self._run("a")
        self._run("b")
        self._run("a")   # 触碰 a，b 成为最久未用
        self._run("c")
        self.assertEqual(_sig_keys(self.cache), ["a", "c"])

    def test_eviction_acquires_entry_lock(self):
        """淘汰必须等在途回放结束（与 _release_caches 同标准）。"""
        self._run("a")
        self._run("b")
        self._run("a")
        # 手动占住将被淘汰对象的锁：run 若不等锁，会在锁被占用时仍完成淘汰
        evict_me = next(e for k, e in self.cache.entries.items() if k[0] == "b")
        order = []
        with evict_me.lock:
            def release():
                order.append("released")
                evict_me.lock.release()
            timer = threading.Timer(0.2, release)
            evict_me.lock.acquire()  # 故意多占一层（RLock 计数）
            timer.start()
            try:
                self._run("c")
                order.append("evicted")
            finally:
                timer.cancel()
                evict_me.lock.release()
        self.assertEqual(_sig_keys(self.cache), ["a", "c"])


class TestRunCudaGraphFallback(unittest.TestCase):
    def test_cpu_device_runs_eager_without_cache(self):
        class Owner:
            pass

        owner = Owner()
        out = run_cuda_graph(owner, "ns", lambda t: t * 2, torch.ones(3))
        self.assertTrue(torch.equal(out, torch.full((3,), 2.0)))
        self.assertFalse(hasattr(owner, "_rvc_cuda_graph_cache"),
                         "非 CUDA 路径不应挂载图缓存")


if __name__ == "__main__":
    unittest.main()
