"""CUDA Graph 运行时配置 — 设备检测与启用开关。

推理层的 CUDA Graph 捕获/回放（rvc.inference.cuda_graph）依赖本模块的
设备判断函数。放在 runtime 层避免反向依赖（runtime 不应依赖 inference）。
"""
import os

import torch

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


def configure_cuda_graph(device):
    """初始化 CUDA Graph 支持。返回是否启用。"""
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
