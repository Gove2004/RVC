"""运行时设备配置。"""
import logging
import threading

import torch

from rvc.runtime.cuda_graph import configure_cuda_graph

logger = logging.getLogger(__name__)


class Config:
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
        self._init_device()

        if configure_cuda_graph(self.device):
            self.use_cuda_graph = True
            logger.info("CUDA Graph 已启用（GPU：%s）", self.gpu_name)
        else:
            self.use_cuda_graph = False

    def _init_device(self) -> None:
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        logger.info("GPU：%s", self.gpu_name)

        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4
        )
