"""从训练特征构建 .pt 特征索引（torch 版，免 faiss/sklearn）。

上游用 faiss IndexIVFFlat（需 MiniBatchKMeans 聚类训练）。本项目检索后端
已改为 GPU 暴力近邻（rvc/inference/index_retrieval.py），暴力近邻无需索引
训练，直接把全部训练帧 L2 归一化后存成一个 .pt 张量即可，加载即用。

内存策略（低配机 / 2GB 内存可跑）：
  早期版本先把全部 npy 累积成 list 再 concatenate + normalize，峰值 ≈ 3×
  特征总量，特征多的实验在低内存云机上会 OOM（Killed）。现改为两遍流式：
    Pass 1  mmap 只读各 npy 的 shape，统计总帧数与维度（不载入数据）
    Pass 2  预分配一块 fp16 (N, D) 缓冲，逐段 load → 就地 L2 归一化 → 写入
  峰值 ≈ 0.5× 特征总量（fp16）+ 单段。L2 归一化是逐行独立运算，分段做与
  整库做数学上完全等价，无精度损失。
  fp16 存储由 load_index 端 .float() 兼容（推理时统一转 fp32 再兜底归一化）。
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_index_from_features(feature_dir, out_path, log=None):
    """扫描特征目录（*.npy，每段一个 (T,768)）→ 流式拼接归一化 → 存 .pt。

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

    # ---- Pass 1: mmap 只读 shape，统计总帧数 / 校验维度（不载入数据）----
    meta = []  # [(path, rows)]
    total = 0
    dim = None
    for name in npy_files:
        path = os.path.join(feature_dir, name)
        try:
            probe = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as e:
            emit(f"[索引] 跳过 {name}: {e}")
            continue
        if probe.ndim == 1:
            rows_this, dim_this = 1, int(probe.shape[0])
        elif probe.ndim == 2:
            rows_this, dim_this = int(probe.shape[0]), int(probe.shape[1])
        else:
            emit(f"[索引] 跳过 {name}: 非 1D/2D 数组（ndim={probe.ndim}）")
            continue
        del probe
        if rows_this == 0 or dim_this == 0:
            continue
        if dim is None:
            dim = dim_this
        elif dim_this != dim:
            emit(f"[索引] 跳过 {name}: 维度 {dim_this} ≠ 其他段 {dim}（可能特征器不一致，请删 3_feature768 重提）")
            continue
        meta.append((path, rows_this))
        total += rows_this

    if not meta:
        emit(f"[索引] 无有效特征可入库: {feature_dir}")
        return 0

    # ---- Pass 2: 预分配 fp16 缓冲，逐段 load → 就地归一化 → 写入 ----
    buf = np.empty((total, dim), dtype=np.float16)
    pos = 0
    for path, rows_this in meta:
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:  # 兜底：单帧存成一维
            arr = arr.reshape(1, -1)
        arr = arr[:rows_this]
        # 逐行 L2 归一化（行内独立，与整库归一化等价）
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        arr /= norms
        buf[pos:pos + rows_this] = arr.astype(np.float16)
        pos += rows_this
        del arr
        if pos % 20000 < rows_this or pos == total:
            emit(f"[索引] 处理中 {pos}/{total} 帧")

    t = torch.from_numpy(buf)  # (N, D) fp16，共享内存零拷贝
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(t, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    emit(f"[索引] 已构建 {total} 帧 × {dim} 维（fp16）→ {out_path}（{size_mb:.1f} MB）")
    return total
