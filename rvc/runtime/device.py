"""运行时设备配置。"""
import logging
import threading

import torch

from rvc.runtime.graph import RuntimeOptions, configure_cuda_graph

logger = logging.getLogger(__name__)


class RuntimeDeviceConfig:
    """全局设备配置"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = "cuda:0"
        self.is_half = True
        self.use_cuda_graph = False
        self.gpu_name = None
        self.gpu_mem = None
        # 显式运行时选项：取代旧版 RVC_CUDA_GRAPH* 环境变量通道
        self.graph_options = RuntimeOptions()
        self._init_device()

        if configure_cuda_graph(self.device, self.graph_options):
            self.use_cuda_graph = True
            logger.info("CUDA Graph 已启用（GPU：%s）", self.gpu_name)
        else:
            self.use_cuda_graph = False

    def _init_device(self) -> None:
        # 现状契约：无 CUDA 环境直接失败，不做 CPU 回退。
        # 显式抛出可读错误（而非让 torch 裸异常穿透），GUI 层据此弹友好提示。
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA GPU（torch.cuda.is_available() = False）。\n"
                "本程序需要 CUDA 环境运行：请确认已安装 NVIDIA 显卡驱动，"
                "且当前 PyTorch 为 CUDA 构建版本。"
            )
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        logger.info("GPU：%s", self.gpu_name)

        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4
        )
