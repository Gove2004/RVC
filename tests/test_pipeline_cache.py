"""pipeline InferenceCache LRU 语义契约测试。

锁定现状行为（P1-10）：仅容量淘汰，无 mtime/内容校验；
各槽位容量 hubert=2 / rmvpe=1 / fcpe=1 / synthesizer=2。
"""
import unittest

from rvc.pipeline.cache import InferenceCache


class _Dummy:
    pass


class TestInferenceCacheLru(unittest.TestCase):
    def test_synthesizer_capacity_two_and_lru_order(self):
        cache = InferenceCache()
        a, b, c = _Dummy(), _Dummy(), _Dummy()
        cache.set_synthesizer("a.pth", a)
        cache.set_synthesizer("b.pth", b)
        # 命中 a → b 成为最久未用
        self.assertIs(cache.get_synthesizer("a.pth"), a)
        cache.set_synthesizer("c.pth", c)  # 淘汰 b
        self.assertIsNone(cache.get_synthesizer("b.pth"))
        self.assertIs(cache.get_synthesizer("a.pth"), a)
        self.assertIs(cache.get_synthesizer("c.pth"), c)

    def test_single_slot_eviction(self):
        """rmvpe/fcpe 槽位容量 1：第二个 key 进入即淘汰第一个。"""
        cache = InferenceCache()
        first, second = _Dummy(), _Dummy()
        cache.set_rmvpe(("cpu", False), first)
        cache.set_rmvpe(("cuda:0", False), second)
        self.assertIsNone(cache.get_rmvpe(("cpu", False)))
        self.assertIs(cache.get_rmvpe(("cuda:0", False)), second)

    def test_same_key_overwrite(self):
        """同 key 重设：原地更新值，不触发淘汰。"""
        cache = InferenceCache()
        first, second = _Dummy(), _Dummy()
        cache.set_rmvpe(("cpu", False), first)
        cache.set_rmvpe(("cpu", False), second)
        self.assertIs(cache.get_rmvpe(("cpu", False)), second)

    def test_no_mtime_invalidation(self):
        """同名 pth 覆盖后仍返回旧缓存对象（现状契约，无文件校验）。"""
        cache = InferenceCache()
        old = _Dummy()
        cache.set_synthesizer("same.pth", old)
        self.assertIs(cache.get_synthesizer("same.pth"), old)

    def test_synthesizer_bundles_snapshot(self):
        cache = InferenceCache()
        a, b = _Dummy(), _Dummy()
        cache.set_synthesizer("a.pth", a)
        cache.set_synthesizer("b.pth", b)
        bundles = cache.synthesizer_bundles()
        self.assertEqual(len(bundles), 2)
        self.assertIn(a, bundles)

    def test_clear_graph_helpers_noop_without_real_models(self):
        """无 CUDA Graph 的哑对象上调用清除方法必须安全。"""
        cache = InferenceCache()
        cache.set_synthesizer("a.pth", _Dummy())
        cache.clear_synthesizer_cuda_graphs()
        cache.clear_f0_cuda_graph_caches()


if __name__ == "__main__":
    unittest.main()
