"""InferenceCache 深度测试 — LRU 边界 / 线程安全 / 各槽位淘汰 / CUDA Graph 清除。

在 test_inference_cache.py 基础上补充更深入的测试用例。
"""
import threading
import time
import unittest

from rvc.inference.inference_cache import InferenceCache, _LRU


class TestLRUDeep(unittest.TestCase):
    """_LRU 边界条件和线程安全测试。"""

    def test_maxsize_one(self):
        """maxsize=1 时只保留最近一个。"""
        lru = _LRU(1)
        lru.set("a", 1)
        self.assertEqual(lru.get("a"), 1)
        lru.set("b", 2)
        self.assertIsNone(lru.get("a"))
        self.assertEqual(lru.get("b"), 2)

    def test_update_existing_key(self):
        """更新已有 key 会刷新其 LRU 位置。"""
        lru = _LRU(2)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.set("a", 10)  # 更新 a，使其变为最近使用
        lru.set("c", 3)  # 应淘汰 b
        self.assertEqual(lru.get("a"), 10)
        self.assertIsNone(lru.get("b"))
        self.assertEqual(lru.get("c"), 3)

    def test_get_none_value(self):
        """value 为 None 时 get 返回 None，但 key 仍在字典中。"""
        lru = _LRU(2)
        lru.set("a", None)
        self.assertIsNone(lru.get("a"))
        # 不触发 move_to_end（因为 value is None），但 key 仍在
        lru.set("b", 2)
        lru.set("c", 3)  # maxsize=2，应淘汰最久未用的

    def test_values_empty(self):
        """空 LRU 的 values() 返回空列表。"""
        lru = _LRU(3)
        self.assertEqual(lru.values(), [])

    def test_values_order(self):
        """values() 按 LRU 顺序返回（最久未用到最近使用）。"""
        lru = _LRU(3)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.set("c", 3)
        values = lru.values()
        self.assertEqual(values, [1, 2, 3])

    def test_thread_safety_concurrent_reads(self):
        """多线程并发读不崩溃。"""
        lru = _LRU(100)
        for i in range(100):
            lru.set(f"key_{i}", i)

        errors = []
        def reader():
            try:
                for _ in range(1000):
                    lru.get("key_0")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_thread_safety_concurrent_writes(self):
        """多线程并发写不崩溃，最终元素数不超过 maxsize。"""
        lru = _LRU(50)
        errors = []

        def writer(tid):
            try:
                for i in range(100):
                    lru.set(f"t{tid}_k{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(len(lru.values()), 50)

    def test_thread_safety_read_write_mixed(self):
        """读写混合并发不崩溃。"""
        lru = _LRU(20)
        for i in range(20):
            lru.set(f"key_{i}", i)

        errors = []

        def reader():
            try:
                for _ in range(500):
                    lru.get("key_0")
                    lru.values()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(500):
                    lru.set(f"new_key_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


class TestInferenceCacheDeep(unittest.TestCase):
    """InferenceCache 各槽位 LRU 淘汰和集成测试。"""

    def test_hubert_eviction(self):
        """HuBERT 槽位 maxsize=2，第 3 个淘汰最久未用的。"""
        cache = InferenceCache()
        cache.set_hubert(("cuda", True, "base"), "hubert_base")
        cache.set_hubert(("cuda", True, "chinese"), "hubert_chinese")
        cache.set_hubert(("cuda", False, "base"), "hubert_base_fp32")
        # 应淘汰第一个（最久未用）
        self.assertIsNone(cache.get_hubert(("cuda", True, "base")))
        self.assertEqual(cache.get_hubert(("cuda", True, "chinese")), "hubert_chinese")
        self.assertEqual(cache.get_hubert(("cuda", False, "base")), "hubert_base_fp32")

    def test_hubert_access_refreshes(self):
        """访问 HuBERT 刷新 LRU 位置。"""
        cache = InferenceCache()
        cache.set_hubert(("cuda", True, "base"), "hubert_base")
        cache.set_hubert(("cuda", True, "chinese"), "hubert_chinese")
        cache.get_hubert(("cuda", True, "base"))  # 刷新 base
        cache.set_hubert(("cuda", False, "base"), "hubert_base_fp32")
        # 应淘汰 chinese（最久未用）
        self.assertEqual(cache.get_hubert(("cuda", True, "base")), "hubert_base")
        self.assertIsNone(cache.get_hubert(("cuda", True, "chinese")))

    def test_rmvpe_eviction(self):
        """RMVPE 槽位 maxsize=1，第 2 个淘汰第 1 个。"""
        cache = InferenceCache()
        cache.set_rmvpe(("cuda", True), "rmvpe_fp16")
        cache.set_rmvpe(("cuda", False), "rmvpe_fp32")
        self.assertIsNone(cache.get_rmvpe(("cuda", True)))
        self.assertEqual(cache.get_rmvpe(("cuda", False)), "rmvpe_fp32")

    def test_fcpe_eviction(self):
        """FCPE 槽位 maxsize=1，第 2 个淘汰第 1 个。"""
        cache = InferenceCache()
        cache.set_fcpe("cuda_fp16", "fcpe_fp16")
        cache.set_fcpe("cuda_fp32", "fcpe_fp32")
        self.assertIsNone(cache.get_fcpe("cuda_fp16"))
        self.assertEqual(cache.get_fcpe("cuda_fp32"), "fcpe_fp32")

    def test_synthesizer_eviction(self):
        """Synthesizer 槽位 maxsize=2，第 3 个淘汰最久未用的。"""
        cache = InferenceCache()
        cache.set_synthesizer("model_a.pth", "syn_a")
        cache.set_synthesizer("model_b.pth", "syn_b")
        cache.set_synthesizer("model_c.pth", "syn_c")
        self.assertIsNone(cache.get_synthesizer("model_a.pth"))
        self.assertEqual(cache.get_synthesizer("model_b.pth"), "syn_b")
        self.assertEqual(cache.get_synthesizer("model_c.pth"), "syn_c")

    def test_synthesizer_access_refreshes(self):
        """访问 Synthesizer 刷新 LRU 位置。"""
        cache = InferenceCache()
        cache.set_synthesizer("model_a.pth", "syn_a")
        cache.set_synthesizer("model_b.pth", "syn_b")
        cache.get_synthesizer("model_a.pth")  # 刷新 a
        cache.set_synthesizer("model_c.pth", "syn_c")
        # 应淘汰 b
        self.assertEqual(cache.get_synthesizer("model_a.pth"), "syn_a")
        self.assertIsNone(cache.get_synthesizer("model_b.pth"))
        self.assertEqual(cache.get_synthesizer("model_c.pth"), "syn_c")

    def test_independent_caches(self):
        """各槽位独立，互不影响。"""
        cache = InferenceCache()
        cache.set_hubert("key", "hubert_val")
        cache.set_rmvpe("key", "rmvpe_val")
        cache.set_fcpe("key", "fcpe_val")
        cache.set_synthesizer("key", "syn_val")

        self.assertEqual(cache.get_hubert("key"), "hubert_val")
        self.assertEqual(cache.get_rmvpe("key"), "rmvpe_val")
        self.assertEqual(cache.get_fcpe("key"), "fcpe_val")
        self.assertEqual(cache.get_synthesizer("key"), "syn_val")

    def test_clear_f0_cuda_graph_empty(self):
        """空缓存时 clear_f0_cuda_graph_caches 不崩溃。"""
        cache = InferenceCache()
        cache.clear_f0_cuda_graph_caches()  # 不应抛异常

    def test_default_cache_is_instance(self):
        """default_inference_cache 是 InferenceCache 实例。"""
        from rvc.inference.inference_cache import default_inference_cache
        self.assertIsInstance(default_inference_cache, InferenceCache)


if __name__ == "__main__":
    unittest.main()
