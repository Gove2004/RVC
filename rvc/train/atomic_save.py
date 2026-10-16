"""训练中间产物的原子 npy 写入 — 中断不留截断文件。

续跑逻辑按「文件存在即跳过」判定，非原子写在中途强杀（Ctrl+C 强退/断电）时
会留下写到一半的截断 .npy，下次被判定「已完成」静默复用——损坏特征混入训练
轻则该样本 loss 异常，重则 NaN 传染。先写 .tmp 再 os.replace（与 checkpoint /
config 的原子写同一口径）保证任何时刻磁盘上的 .npy 都是完整的。
"""
import os
from pathlib import Path

import numpy as np


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> None:
    """原子写 npy：写 .tmp 后 os.replace 到目标路径。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)
