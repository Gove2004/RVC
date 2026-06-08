import json
import os
import shutil
import sys
import logging

import torch

logger = logging.getLogger(__name__)

version_config_list = [
    "v2/48k.json",
    "v2/32k.json",
]


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = "cuda:0"
        self.is_half = True
        self.gpu_name = None
        self.json_config = self._load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self._device_config()

    @staticmethod
    def _load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    def _device_config(self) -> tuple:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This project requires an NVIDIA GPU.")
            sys.exit(1)

        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        logger.info("GPU: %s", self.gpu_name)

        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory
            / 1024
            / 1024
            / 1024
            + 0.4
        )

        x_pad = 3
        x_query = 10
        x_center = 60
        x_max = 65

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
