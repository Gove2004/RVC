"""从训练特征构建 .pt 特征索引（torch 版，免 faiss/sklearn）。

上游用 faiss IndexIVFFlat（需 MiniBatchKMeans 聚类训练）。本项目检索后端
已改为 GPU 暴力近邻（rvc/inference/index_retrieval.py），暴力近邻无需索引
训练，直接把全部训练帧 L2 归一化后存成一个 .pt 张量即可，加载即用。
"""
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_index_from_features(feature_dir, out_path, log=None):
    """扫描特征目录（*.npy，每段一个 (T,768)）→ 拼接归一化 → 存 .pt。

    Args:
        feature_dir: 训练产物目录，如 <exp>/3_feature768
        out_path:    输出 .pt 文件路径
        log:         可选日志回调 callable(str)

    Returns:
        int: 入库帧数；目录为空/无特征时返回 0 且不写文件
    """
    def emit(msg):
        (log or print)(msg)

    feature_dir = os.path.abspath(feature_dir)
    if not os.path.isdir(feature_dir):
        emit(f"[索引] 特征目录不存在: {feature_dir}")
        return 0

    npy_files = sorted(
        f for f in os.listdir(feature_dir) if f.lower().endswith(".npy")
    )
    if not npy_files:
        emit(f"[索引] 特征目录为空: {feature_dir}")
        return 0

    rows = []
    for name in npy_files:
        try:
            arr = np.load(os.path.join(feature_dir, name), allow_pickle=False)
        except Exception as e:
            emit(f"[索引] 跳过 {name}: {e}")
            continue
        if arr.ndim == 1:  # 兜底：单帧存成一维
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] == 0:
            continue
        rows.append(arr.astype(np.float32))

    if not rows:
        emit(f"[索引] 无有效特征可入库: {feature_dir}")
        return 0

    refs = np.concatenate(rows, axis=0)  # (N, 768)
    del rows
    n = refs.shape[0]
    # 全量归一化（CPU 一次性，几万帧无压力）
    t = torch.from_numpy(refs)
    t = F.normalize(t.float(), dim=1)
    t = t.contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(t, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    emit(f"[索引] 已构建 {n} 帧 → {out_path}（{size_mb:.1f} MB）")
    return n
