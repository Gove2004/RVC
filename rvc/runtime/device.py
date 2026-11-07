"""运行时设备配置。"""
import ctypes
import glob
import logging
import os
import threading

import torch

from rvc.runtime.graph import RuntimeOptions, configure_cuda_graph

logger = logging.getLogger(__name__)


def _prefer_blocking_sync() -> None:
    """让 CUDA 同步阻塞睡眠而非自旋，省音频回调线程等待 GPU 时的 CPU。

    实测（RTX 5060 Laptop，真实链路每块）：自旋 CPU 19ms ≈ 墙钟 20ms；改阻塞后
    CPU 降到 ~4ms，墙钟不变。代价是唤醒依赖 OS 调度器（健康系统 <1ms），重负载
    下的实时可靠性需实听验证。

    必须在 CUDA 上下文创建前调用，进程级生效。要加载 torch 自带的那份 cudart
    （与 torch 同一运行时实例，flags 才落在 torch 用的上下文上）；失败仅告警，
    退回自旋不影响正确性。
    """
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    candidates = (
        glob.glob(os.path.join(lib_dir, "cudart64_*.dll"))
        + glob.glob(os.path.join(lib_dir, "libcudart.so*"))
    )
    if not candidates:
        logger.warning("未找到 torch 自带 cudart，CUDA 同步保持自旋（CPU 占用较高）")
        return
    try:
        cudart = ctypes.CDLL(candidates[0])
        rc = cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))  # cudaDeviceScheduleBlockingSync
        if rc != 0:
            logger.warning("cudaSetDeviceFlags 返回 %d，CUDA 同步保持自旋", rc)
    except Exception as exc:
        logger.warning("设置 CUDA 阻塞同步失败，保持自旋：%s", exc)


class RuntimeDeviceConfig:
    """全局设备配置（进程内单例）。

    `device_index` 只在**首次构造**时生效（默认 0）：模型权重与 CUDA Graph 都按
    当时的设备捕获/驻留，运行中换卡需要迁移全部模型，因此多卡切换的契约是
    「改配置 → 重启程序」，本类不做热切换。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, device_index: int = 0) -> "RuntimeDeviceConfig":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._init(device_index)
                cls._instance = instance
        return cls._instance

    def _init(self, device_index: int) -> None:
        self.is_half = True
        self.gpu_name = None
        self._init_device(device_index)

        if configure_cuda_graph(self.device):
            logger.info("CUDA Graph 已启用（GPU：%s）", self.gpu_name)

    def _init_device(self, device_index: int) -> None:
        # 阻塞同步须在上下文创建前设（见 _prefer_blocking_sync）：省回调线程自旋 CPU。
        _prefer_blocking_sync()
        # 现状契约：无 CUDA 环境直接失败，不做 CPU 回退。
        # 显式抛出可读错误（而非让 torch 裸异常穿透），GUI 层据此弹友好提示。
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA GPU（torch.cuda.is_available() = False）。\n"
                "本程序需要 CUDA 环境运行：请确认已安装 NVIDIA 显卡驱动，"
                "且当前 PyTorch 为 CUDA 构建版本。"
            )
        device_count = torch.cuda.device_count()
        index = int(device_index)
        if not 0 <= index < device_count:
            logger.warning(
                "配置的 GPU 编号 %d 超出范围（本机 %d 张卡），回退到 GPU 0", index, device_count,
            )
            index = 0
        self.device_index = index
        self.device = f"cuda:{index}"
        self.gpu_name = torch.cuda.get_device_name(index)
        logger.info("GPU：%s", self.gpu_name)
