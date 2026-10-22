"""训练中间产物的原子写入 — 通用骨架在 rvc.io.atomic，这里只留数据格式。

续跑逻辑按「文件存在即跳过」判定，非原子写在中途强杀（Ctrl+C 强退/断电）时
会留下写到一半的截断文件，下次被判定「已完成」静默复用——损坏特征混入训练
轻则该样本 loss 异常，重则 NaN 传染。
"""
from pathlib import Path
from typing import Any

import numpy as np

from rvc.io.atomic import atomic_write


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> None:
    """原子写 npy。"""
    atomic_write(path, lambda f: np.save(f, arr, allow_pickle=False))


def save_torch_atomic(path: str | Path, obj: Any) -> None:
    """原子写 torch 序列化文件（.pt 等）。"""
    import torch

    atomic_write(path, lambda f: torch.save(obj, f))
