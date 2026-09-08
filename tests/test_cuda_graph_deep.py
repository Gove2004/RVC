"""CUDA Graph 深度测试 — 纯函数部分（克隆输出/张量签名/缓存管理）。

覆盖：
- _clone_output: tensor/tuple/其他类型的深拷贝
- _tensor_signature: 形状/步长/dtype/device/requires_grad
- _GraphCache: 初始化、max_entries、统计计数、LRU 淘汰
- run_cuda_graph: 无输入直接调用、CPU 设备直接调用
- clear_cuda_graph_cache: 无缓存不崩溃、有缓存清除
"""
import os
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import torch

from rvc.inference.cuda_graph import (
    MAX_CACHE_ENV,
    _clone_output,
    _GraphCache,
    _tensor_signature,
    clear_cuda_graph_cache,
    run_cuda_graph,
)


class TestCloneOutput(unittest.TestCase):
    """_clone_output 测试。"""

    def test_clone_tensor(self):
        """张量被克隆（不共享内存）。"""
        t = torch.tensor([1.0, 2.0, 3.0])
        cloned = _clone_output(t)
        self.assertIsNot(cloned, t)
        self.assertTrue(torch.equal(cloned, t))
        # 修改原张量不影响克隆
        t[0] = 999.0
        self.assertEqual(cloned[0].item(), 1.0)

    def test_clone_tuple_of_tensors(self):
        """元组中的张量被克隆。"""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        result = _clone_output((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsNot(result[0], t1)
        self.assertIsNot(result[1], t2)
        self.assertTrue(torch.equal(result[0], t1))
        self.assertTrue(torch.equal(result[1], t2))

    def test_clone_nested_tuple(self):
        """嵌套元组被递归克隆。"""
        t = torch.tensor([1.0])
        result = _clone_output(((t,), (t,)))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tuple)
        self.assertIsNot(result[0][0], t)

    def test_non_tensor_passthrough(self):
        """非张量/元组类型直接返回。"""
        for value in [42, "hello", 3.14, None, [1, 2, 3], {"a": 1}]:
            result = _clone_output(value)
            self.assertIs(result, value)

    def test_empty_tuple(self):
        result = _clone_output(())
        self.assertEqual(result, ())

    def test_tuple_with_mixed_types(self):
        t = torch.tensor([1.0])
        result = _clone_output((t, "hello", 42))
        self.assertEqual(len(result), 3)
        self.assertIsNot(result[0], t)
        self.assertEqual(result[1], "hello")
        self.assertEqual(result[2], 42)

    def test_clone_preserves_dtype(self):
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            t = torch.zeros(3, dtype=dtype)
            cloned = _clone_output(t)
            self.assertEqual(cloned.dtype, dtype)

    def test_clone_preserves_shape(self):
        for shape in [(3,), (2, 3), (1, 2, 3), (5,)]:
            t = torch.zeros(shape)
            cloned = _clone_output(t)
            self.assertEqual(cloned.shape, t.shape)


class TestTensorSignature(unittest.TestCase):
    """_tensor_signature 测试。"""

    def test_returns_tuple(self):
        t = torch.zeros(3)
        sig = _tensor_signature(t)
        self.assertIsInstance(sig, tuple)
        self.assertEqual(len(sig), 5)

    def test_shape_in_signature(self):
        t = torch.zeros(2, 3)
        sig = _tensor_signature(t)
        self.assertEqual(sig[0], (2, 3))

    def test_stride_in_signature(self):
        t = torch.zeros(2, 3)
        sig = _tensor_signature(t)
        self.assertEqual(sig[1], tuple(t.stride()))

    def test_dtype_in_signature(self):
        t = torch.zeros(3, dtype=torch.float64)
        sig = _tensor_signature(t)
        self.assertIn("float64", sig[2])

    def test_device_in_signature(self):
        t = torch.zeros(3)
        sig = _tensor_signature(t)
        self.assertIn("cpu", sig[3])

    def test_requires_grad_in_signature(self):
        t = torch.zeros(3, requires_grad=True)
        sig = _tensor_signature(t)
        self.assertTrue(sig[4])

    def test_different_tensors_different_signatures(self):
        t1 = torch.zeros(3)
        t2 = torch.zeros(4)
        self.assertNotEqual(_tensor_signature(t1), _tensor_signature(t2))

    def test_same_tensors_same_signatures(self):
        t1 = torch.zeros(3, dtype=torch.float32)
        t2 = torch.zeros(3, dtype=torch.float32)
        self.assertEqual(_tensor_signature(t1), _tensor_signature(t2))

    def test_different_dtype_different_signatures(self):
        t1 = torch.zeros(3, dtype=torch.float32)
        t2 = torch.zeros(3, dtype=torch.float64)
        self.assertNotEqual(_tensor_signature(t1), _tensor_signature(t2))

    def test_hashable(self):
        """签名可以作为字典键。"""
        t = torch.zeros(3)
        sig = _tensor_signature(t)
        d = {sig: "value"}
        self.assertEqual(d[sig], "value")

    def test_various_shapes(self):
        for shape in [(1,), (10,), (100,), (1, 10), (10, 10), (1, 10, 10)]:
            t = torch.zeros(shape)
            sig = _tensor_signature(t)
            self.assertEqual(sig[0], shape)


class TestGraphCacheInit(unittest.TestCase):
    """_GraphCache 初始化测试。"""

    def test_creates_ordered_dict(self):
        cache = _GraphCache()
        self.assertIsInstance(cache.entries, OrderedDict)

    def test_initial_counts_zero(self):
        cache = _GraphCache()
        self.assertEqual(cache.capture_count, 0)
        self.assertEqual(cache.replay_count, 0)
        self.assertEqual(cache.eviction_count, 0)
        self.assertEqual(cache.capture_ms, 0.0)

    def test_default_max_entries_8(self):
        cache = _GraphCache()
        self.assertEqual(cache.max_entries, 8)

    def test_max_entries_from_env(self):
        with patch.dict(os.environ, {MAX_CACHE_ENV: "16"}):
            cache = _GraphCache()
            self.assertEqual(cache.max_entries, 16)

    def test_max_entries_minimum_1(self):
        with patch.dict(os.environ, {MAX_CACHE_ENV: "0"}):
            cache = _GraphCache()
            self.assertEqual(cache.max_entries, 1)

    def test_has_lock(self):
        cache = _GraphCache()
        self.assertIsNotNone(cache.lock)


class TestRunCudaGraph(unittest.TestCase):
    """run_cuda_graph 测试（不需要真实 CUDA 设备）。"""

    def test_no_inputs_calls_function_directly(self):
        """无输入时直接调用函数（不走 CUDA Graph）。"""
        owner = MagicMock()
        func = MagicMock(return_value="result")
        result = run_cuda_graph(owner, "test", func)
        self.assertEqual(result, "result")
        func.assert_called_once()

    def test_cpu_device_calls_function_directly(self):
        """CPU 设备上直接调用函数（cuda_graph_enabled 返回 False）。"""
        owner = MagicMock()
        func = MagicMock(return_value="result")
        inp = torch.zeros(3)
        result = run_cuda_graph(owner, "test", func, inp)
        self.assertEqual(result, "result")
        func.assert_called_once_with(inp)

    def test_multiple_cpu_inputs(self):
        owner = MagicMock()
        func = MagicMock(return_value="result")
        inp1 = torch.zeros(3)
        inp2 = torch.zeros(4)
        result = run_cuda_graph(owner, "test", func, inp1, inp2)
        self.assertEqual(result, "result")
        func.assert_called_once_with(inp1, inp2)

    def test_no_cache_created_on_cpu(self):
        """CPU 设备上不创建缓存。"""
        owner = MagicMock(spec=[])
        func = MagicMock(return_value="result")
        inp = torch.zeros(3)
        run_cuda_graph(owner, "test", func, inp)
        self.assertFalse(hasattr(owner, "_rvc_cuda_graph_cache"))

    def test_namespace_passed_to_function(self):
        """namespace 不直接传给函数，只作为缓存键。"""
        owner = MagicMock()
        func = MagicMock(return_value="result")
        inp = torch.zeros(3)
        run_cuda_graph(owner, "my-namespace", func, inp)
        func.assert_called_once_with(inp)


class TestClearCudaGraphCache(unittest.TestCase):
    """clear_cuda_graph_cache 测试。"""

    def test_no_cache_no_crash(self):
        """对象没有缓存时不崩溃。"""
        owner = MagicMock(spec=[])
        clear_cuda_graph_cache(owner)  # 不应崩溃

    def test_clears_existing_cache(self):
        """有缓存时清除 entries 并删除属性。"""
        owner = MagicMock()
        cache = _GraphCache()
        cache.entries["key"] = "value"
        owner._rvc_cuda_graph_cache = cache
        clear_cuda_graph_cache(owner)
        self.assertEqual(len(cache.entries), 0)
        self.assertFalse(hasattr(owner, "_rvc_cuda_graph_cache"))

    def test_called_with_none(self):
        """owner 为 None 时不崩溃（getattr 返回 None）。"""
        clear_cuda_graph_cache(None)  # 不应崩溃

    def test_multiple_calls_no_crash(self):
        """多次调用不崩溃。"""
        owner = MagicMock()
        clear_cuda_graph_cache(owner)
        clear_cuda_graph_cache(owner)
        clear_cuda_graph_cache(owner)


class TestCudaGraphConstants(unittest.TestCase):
    """CUDA Graph 常量测试。"""

    def test_max_cache_env_is_string(self):
        self.assertIsInstance(MAX_CACHE_ENV, str)
        self.assertGreater(len(MAX_CACHE_ENV), 0)

    def test_env_name_imported(self):
        from rvc.runtime.cuda_graph import ENV_NAME
        self.assertIsInstance(ENV_NAME, str)


if __name__ == "__main__":
    unittest.main()
