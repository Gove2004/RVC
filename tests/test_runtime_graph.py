"""runtime.graph 选项下发与清理契约测试。

现状契约：
- 图开关只由 RuntimeOptions 显式下发（环境变量不再参与）；
- 无图环境（cpu / 未配置）时 run_cuda_graph 直通原函数；
- clear/purge 后 owner 不再持有缓存，且可安全重复调用。
"""
import unittest
from unittest.mock import patch

import torch

from rvc.runtime import graph


def _requires_cuda(test):
    return unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")(test)


class TestRuntimeOptions(unittest.TestCase):
    def tearDown(self):
        # 恢复"未配置"状态，避免污染同进程其他用例
        graph._active_options = None

    def test_cpu_device_never_enables(self):
        enabled = graph.configure_cuda_graph("cpu", graph.RuntimeOptions())
        self.assertFalse(enabled)
        self.assertFalse(graph.cuda_graph_enabled("cpu"))

    def test_env_var_no_longer_consulted(self):
        """旧环境变量通道已删除：置 0 也不得影响显式选项。"""
        import os

        os.environ["RVC_CUDA_GRAPH"] = "0"
        try:
            enabled = graph.configure_cuda_graph("cuda:0", graph.RuntimeOptions())
            self.assertEqual(enabled, torch.cuda.is_available())
        finally:
            os.environ.pop("RVC_CUDA_GRAPH", None)

    def test_disabled_option_disables_graph(self):
        graph.configure_cuda_graph(
            "cuda:0", graph.RuntimeOptions(use_cuda_graph=False)
        )
        self.assertFalse(graph.cuda_graph_enabled("cuda:0"))


class TestGraphBypass(unittest.TestCase):
    def tearDown(self):
        graph._active_options = None

    def test_unconfigured_runs_eager(self):
        graph._active_options = None
        calls = []

        def fn(x):
            calls.append(1)
            return x * 2

        out = graph.run_cuda_graph(self, "ns", fn, torch.zeros(4))
        self.assertTrue(torch.equal(out, torch.zeros(4) * 2))
        self.assertEqual(len(calls), 1)
        self.assertFalse(hasattr(self, "_rvc_cuda_graph_cache"))

    def test_purge_without_cache_is_noop(self):
        graph.purge_cuda_graphs(self)  # 无缓存 owner：安全、无异常
        self.assertFalse(hasattr(self, "_rvc_cuda_graph_cache"))


class TestCaptureAndPurge(unittest.TestCase):
    def tearDown(self):
        graph.purge_cuda_graphs(self)
        graph._active_options = None

    @_requires_cuda
    def test_capture_replay_and_purge(self):
        graph.configure_cuda_graph("cuda:0", graph.RuntimeOptions())
        device = torch.device("cuda:0")

        def fn(x):
            return (x * 2 + 1).clamp(min=0)

        x1 = torch.ones(8, device=device)
        out1 = graph.run_cuda_graph(self, "ns", fn, x1)
        self.assertTrue(torch.equal(out1, fn(x1).cpu().to(device)))
        self.assertTrue(hasattr(self, "_rvc_cuda_graph_cache"))
        cache = self._rvc_cuda_graph_cache
        self.assertEqual(cache.capture_count, 1)
        self.assertEqual(cache.replay_count, 1)

        # 同签名第二次调用走回放：不再新增 capture
        out2 = graph.run_cuda_graph(self, "ns", fn, x1 + 1)
        self.assertTrue(torch.equal(out2, fn(x1 + 1)))
        self.assertEqual(cache.capture_count, 1)
        self.assertEqual(cache.replay_count, 2)

        # purge：缓存摘除 + 条目清空；再次调用重新捕获
        graph.purge_cuda_graphs(self)
        self.assertFalse(hasattr(self, "_rvc_cuda_graph_cache"))
        out3 = graph.run_cuda_graph(self, "ns", fn, x1)
        self.assertTrue(torch.equal(out3, fn(x1)))
        self.assertEqual(self._rvc_cuda_graph_cache.capture_count, 1)

    @_requires_cuda
    def test_purge_multi_owner_single_release(self):
        graph.configure_cuda_graph("cuda:0", graph.RuntimeOptions())
        device = torch.device("cuda:0")
        fn = lambda x: x + 1  # noqa: E731
        owners = [unittest.TestCase(), unittest.TestCase()]
        for owner in owners:
            graph.run_cuda_graph(owner, "ns", fn, torch.zeros(4, device=device))
        caches = [o._rvc_cuda_graph_cache for o in owners]
        graph.purge_cuda_graphs(*owners)
        for owner in owners:
            self.assertFalse(hasattr(owner, "_rvc_cuda_graph_cache"))
        for cache in caches:
            self.assertEqual(len(cache.entries), 0)


if __name__ == "__main__":
    unittest.main()
