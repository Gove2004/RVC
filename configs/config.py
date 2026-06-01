import os
import sys
import json
import shutil

import torch
import logging

logger = logging.getLogger(__name__)

version_config_list = [
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    def device_config(self) -> tuple:
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
