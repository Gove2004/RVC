"""训练中间产物的原子写入 — 中断不留截断文件。

续跑逻辑按「文件存在即跳过」判定，非原子写在中途强杀（Ctrl+C 强退/断电）时
会留下写到一半的截断文件，下次被判定「已完成」静默复用——损坏特征混入训练
轻则该样本 loss 异常，重则 NaN 传染。先写 .tmp 再 os.replace（与 checkpoint /
config 的原子写同一口径）保证任何时刻磁盘上的文件都是完整的。
"""
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_write(path: str | Path, writer: Callable[[Any], None]) -> None:
    """原子写通用骨架：写 .tmp 后 os.replace 到目标路径。

    writer 接收已打开的二进制文件对象，负责序列化。
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> None:
    """原子写 npy。"""
    _atomic_write(path, lambda f: np.save(f, arr, allow_pickle=False))


def save_torch_atomic(path: str | Path, obj: Any) -> None:
    """原子写 torch 序列化文件（.pt 等）。"""
    import torch

    _atomic_write(path, lambda f: torch.save(obj, f))
