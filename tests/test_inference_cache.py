"""InferenceCache 单元测试 — LRU 缓存管理。"""
import pytest
import torch

from rvc.models.inference_cache import InferenceCache, _LRU


class TestLRU:
    """_LRU 线程安全 LRU 字典测试。"""

    def test_get_set(self):
        lru = _LRU(2)
        lru.set("a", 1)
        assert lru.get("a") == 1

    def test_get_missing(self):
        lru = _LRU(2)
        assert lru.get("missing") is None

    def test_eviction(self):
        lru = _LRU(2)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.set("c", 3)  # 应淘汰 "a"
        assert lru.get("a") is None
        assert lru.get("b") == 2
        assert lru.get("c") == 3

    def test_lru_order(self):
        lru = _LRU(2)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.get("a")  # 访问 "a"，使其变为最近使用
        lru.set("c", 3)  # 应淘汰 "b"
        assert lru.get("a") == 1
        assert lru.get("b") is None
        assert lru.get("c") == 3

    def test_values_snapshot(self):
        lru = _LRU(3)
        lru.set("a", 1)
        lru.set("b", 2)
        values = lru.values()
        assert set(values) == {1, 2}


class TestInferenceCache:
    """InferenceCache 各槽位 LRU 测试。"""

    def test_hubert_cache(self):
        cache = InferenceCache()
        cache.set_hubert(("cuda", True, "base"), "hubert_base")
        assert cache.get_hubert(("cuda", True, "base")) == "hubert_base"
        assert cache.get_hubert(("cuda", False, "base")) is None

    def test_rmvpe_cache(self):
        cache = InferenceCache()
        cache.set_rmvpe(("cuda", True), "rmvpe_model")
        assert cache.get_rmvpe(("cuda", True)) == "rmvpe_model"

    def test_fcpe_cache(self):
        cache = InferenceCache()
        cache.set_fcpe("cuda", "fcpe_model")
        assert cache.get_fcpe("cuda") == "fcpe_model"

    def test_synthesizer_cache(self):
        cache = InferenceCache()
        cache.set_synthesizer("model.pth", "syn_bundle")
        assert cache.get_synthesizer("model.pth") == "syn_bundle"

    def test_clear_f0_cuda_graph_no_crash(self):
        """无 F0 提取器时 clear_f0_cuda_graph_caches 不应崩溃。"""
        cache = InferenceCache()
        cache.clear_f0_cuda_graph_caches()  # 不应抛异常
