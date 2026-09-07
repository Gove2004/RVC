"""CUDA Graph 推理加速 — 捕获 Synthesizer/HuBERT/F0 模型的前向传播，跳过 Python 调度开销。"""
import logging
import os
import threading
import time
from collections import OrderedDict

import torch


logger = logging.getLogger(__name__)

ENV_NAME = "RVC_CUDA_GRAPH"
MAX_CACHE_ENV = "RVC_CUDA_GRAPH_MAX_CACHE"


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def _cuda_device(device):
    parsed = device if isinstance(device, torch.device) else torch.device(device)
    if parsed.index is None:
        parsed = torch.device("cuda", torch.cuda.current_device())
    return parsed


def _clone_output(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_output(item) for item in value)
    return value


def configure_cuda_graph(device):
    """初始化 CUDA Graph 支持。"""
    if _device_type(device) != "cuda":
        os.environ[ENV_NAME] = "0"
        return False
    os.environ[ENV_NAME] = "1"
    return True


def cuda_graph_enabled(device):
    """判断 CUDA Graph 是否对给定设备生效。"""
    return (
        os.environ.get(ENV_NAME) == "1"
        and _device_type(device) == "cuda"
        and torch.cuda.is_available()
    )


def _tensor_signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        bool(tensor.requires_grad),
    )


class _CapturedCall:
    """单次 CUDA Graph capture + replay。"""

    def __init__(self, function, inputs):
        started = time.perf_counter()
        self.lock = threading.RLock()
        self.inputs = tuple(torch.empty_like(value) for value in inputs)
        for static, value in zip(self.inputs, inputs):
            static.copy_(value)
        device = self.inputs[0].device
        current = torch.cuda.current_stream(device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.no_grad():
            for _ in range(3):
                output = function(*self.inputs)
        current.wait_stream(warmup)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = function(*self.inputs)
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        self.done_event = None
        del output

    def replay(self, inputs):
        with self.lock:
            stream = torch.cuda.current_stream(self.inputs[0].device)
            if self.done_event is not None:
                stream.wait_event(self.done_event)
            for static, value in zip(self.inputs, inputs):
                static.copy_(value, non_blocking=True)
            self.graph.replay()
            output = _clone_output(self.output)
            self.done_event = torch.cuda.Event(blocking=False)
            self.done_event.record(stream)
            return output


class _GraphCache:
    """按签名缓存 CUDA Graph，支持 LRU 淘汰。"""

    def __init__(self):
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.capture_count = 0
        self.replay_count = 0
        self.eviction_count = 0
        self.capture_ms = 0.0

    def run(self, key, function, inputs):
        signature = key + tuple(_tensor_signature(value) for value in inputs)
        with self.lock:
            entry = self.entries.get(signature)
            if entry is None:
                entry = _CapturedCall(function, inputs)
                self.entries[signature] = entry
                self.capture_count += 1
                self.capture_ms += entry.capture_ms
                max_entries = max(1, int(os.environ.get(MAX_CACHE_ENV, "8")))
                while len(self.entries) > max_entries:
                    self.entries.popitem(last=False)
                    self.eviction_count += 1
            else:
                self.entries.move_to_end(signature)
        output = entry.replay(inputs)
        with self.lock:
            self.replay_count += 1
        return output


def run_cuda_graph(owner, namespace, function, *inputs):
    """CUDA Graph 入口：已缓存则 replay，未缓存则 capture。"""
    if not inputs or not cuda_graph_enabled(inputs[0].device):
        return function(*inputs)
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is None:
        cache = _GraphCache()
        setattr(owner, "_rvc_cuda_graph_cache", cache)
    return cache.run((str(namespace),), function, tuple(inputs))


def clear_cuda_graph_cache(owner):
    """切换模型时清除 CUDA Graph 缓存。"""
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is not None:
        cache.entries.clear()
        delattr(owner, "_rvc_cuda_graph_cache")
