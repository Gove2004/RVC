"""运行时设备配置。"""
import logging
import threading

import torch

from rvc.runtime.graph import RuntimeOptions, configure_cuda_graph

logger = logging.getLogger(__name__)


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
        self.use_cuda_graph = False
        self.gpu_name = None
        # 显式运行时选项：取代旧版 RVC_CUDA_GRAPH* 环境变量通道
        self.graph_options = RuntimeOptions()
        self._init_device(device_index)

        if configure_cuda_graph(self.device, self.graph_options):
            self.use_cuda_graph = True
            logger.info("CUDA Graph 已启用（GPU：%s）", self.gpu_name)
        else:
            self.use_cuda_graph = False

    def _init_device(self, device_index: int) -> None:
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
