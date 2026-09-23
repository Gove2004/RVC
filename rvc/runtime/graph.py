"""CUDA Graph：显式开关 + 捕获/回放 + 按签名 LRU 缓存。

为什么在 runtime 层：models（rmvpe/hubert/synthesizer）和 pipeline（features/
synthesis/pitch）都要用图加速，而依赖规则禁止它们互为依赖——唯一的共同
下层是 runtime。

为什么 purge/clear 要强制 GC + 同步 + empty_cache：旧的 _CapturedCall 持有
CUDAGraph 实例和静态输入/输出张量，仅 delattr 不会立即释放 GPU 资源，Python
GC 可能延迟回收。此时新捕获的图可能复用旧静态张量的内存地址，导致新旧图
数据竞争——表现为快速 stop/start 后声音沙哑或输出全零。

为什么用 RuntimeOptions 显式对象而非环境变量：旧实现以 RVC_CUDA_GRAPH /
RVC_CUDA_GRAPH_MAX_CACHE 环境变量做进程内通信，模块 import 顺序敏感、
可测试性差。开关现在由 device.RuntimeDeviceConfig 持有并一次性下发到本模块
（configure_cuda_graph），调用方一律经 run_cuda_graph/cuda_graph_enabled
读取，不再触碰进程环境。
"""
import gc
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

DEFAULT_GRAPH_CACHE_SIZE = 8


@dataclass(frozen=True)
class RuntimeOptions:
    """CUDA Graph 运行时选项（由入口层显式构造，禁用环境变量通道）。"""
    use_cuda_graph: bool = True
    graph_cache_size: int = DEFAULT_GRAPH_CACHE_SIZE


# 进程内唯一选项实例：仅由 configure_cuda_graph 写入（RuntimeDeviceConfig 初始化时调用一次），
# 其余代码只读。显式单点写入优于散落的环境变量读写。
_active_options: RuntimeOptions | None = None


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def _cuda_device(device):
    parsed = device if isinstance(device, torch.device) else torch.device(device)
    if parsed.index is None:
        parsed = torch.device("cuda", torch.cuda.current_device())
    return parsed


def configure_cuda_graph(device, options: RuntimeOptions | None = None) -> bool:
    """下发运行时选项并探测设备。返回 CUDA Graph 是否实际启用。

    本函数是 _active_options 的唯一写入口；重复调用以最后一次为准。
    """
    global _active_options
    opts = options or RuntimeOptions()
    enabled = (
        opts.use_cuda_graph
        and _device_type(device) == "cuda"
        and torch.cuda.is_available()
    )
    _active_options = RuntimeOptions(
        use_cuda_graph=enabled,
        graph_cache_size=max(1, int(opts.graph_cache_size)),
    )
    return enabled


def cuda_graph_enabled(device) -> bool:
    """判断 CUDA Graph 是否对给定设备生效。"""
    return (
        _active_options is not None
        and _active_options.use_cuda_graph
        and _device_type(device) == "cuda"
        and torch.cuda.is_available()
    )


def _clone_output(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_output(item) for item in value)
    return value


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
        self.max_entries = (
            _active_options.graph_cache_size if _active_options is not None
            else DEFAULT_GRAPH_CACHE_SIZE
        )

    def run(self, key, function, inputs):
        signature = key + tuple(_tensor_signature(value) for value in inputs)
        with self.lock:
            entry = self.entries.get(signature)
            if entry is None:
                entry = _CapturedCall(function, inputs)
                self.entries[signature] = entry
                self.capture_count += 1
                self.capture_ms += entry.capture_ms
                while len(self.entries) > self.max_entries:
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


def _detach_cache(owner) -> "_GraphCache | None":
    """摘下 owner 的图缓存（不回收 GPU 资源，回收统一在 purge 里做）。"""
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is not None:
        delattr(owner, "_rvc_cuda_graph_cache")
    return cache


def _release_caches(caches) -> None:
    """回收已摘下的图缓存：逐个取锁等待在途回放结束，再统一 GC + 同步 + 清缓存。

    取锁是有意的安全点：replay 全程持锁，取到锁即保证没有回放仍在引用
    即将销毁的 CUDAGraph/静态张量——这是旧图与新图内存地址不冲突的前提。
    """
    for cache in caches:
        with cache.lock:
            for entry in list(cache.entries.values()):
                with entry.lock:
                    pass
            cache.entries.clear()
    if not caches:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 释放 CUDA Graph 占用的 GPU 内存，确保新图不会复用旧内存地址
        torch.cuda.empty_cache()


def purge_cuda_graphs(*owners):
    """单入口清理：摘下所有 owner 的图缓存，统一回收一次（gc+同步+empty_cache）。

    多模型切换场景（合成器+HuBERT+RMVPE 各持缓存）请用本函数替代逐个
    clear——回收序列只跑一遍，最终状态与逐个 clear 完全一致。
    """
    caches = [c for o in owners if (c := _detach_cache(o)) is not None]
    _release_caches(caches)


def clear_cuda_graph_cache(owner):
    """清除单个 owner 的 CUDA Graph 缓存（purge_cuda_graphs 的单 owner 形式）。"""
    purge_cuda_graphs(owner)
