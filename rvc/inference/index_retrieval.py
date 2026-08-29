"""FAISS index 加载与特征混合。"""
import logging
import os

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_index(index_path: str, inference_cache):
    if not index_path or not os.path.exists(index_path):
        return None, None

    cached = inference_cache.get_index(index_path)
    if cached:
        logger.info("加载特征索引 %s（缓存）", os.path.basename(index_path))
        return cached["index"], cached["index_vectors"]

    index = faiss.read_index(index_path)
    index_vectors = index.reconstruct_n(0, index.ntotal)
    inference_cache.set_index(index_path, {
        "index": index,
        "index_vectors": index_vectors,
    })
    logger.info("加载特征索引 %s", os.path.basename(index_path))
    return index, index_vectors


def faiss_blend(feats_npy: np.ndarray, index, index_vectors: np.ndarray, index_rate: float, is_half: bool) -> np.ndarray:
    k = min(8, index.ntotal)
    score, ix = index.search(feats_npy, k=k)
    if not (ix >= 0).all():
        return feats_npy
    weight = np.square(1.0 / np.maximum(score, 1e-10))
    weight /= weight.sum(axis=1, keepdims=True)
    blended = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
    result = blended * index_rate + feats_npy * (1 - index_rate)
    if is_half:
        result = result.astype("float16")
    return result


def apply_faiss_index(
    feats: torch.Tensor,
    index,
    index_vectors: np.ndarray | None,
    index_rate: float,
    is_half: bool,
    device: str,
    skip_head: int = 0,
    blend_every_n: int = 1,
    blend_counter: list[int] | None = None,
    blend_cache: list | None = None,
) -> torch.Tensor:
    """Apply FAISS index blending with optional frequency reduction.

    降频语义：每 blend_every_n 块真正跑一次 FAISS（含 GPU→CPU→GPU 同步，
    回调内最贵的操作之一）；中间块**沿用上一次的混合结果**（而非原生特征），
    避免音色每 N 块周期性跳动。blend_cache 由调用方持有（如 [None]）。

    Args:
        blend_every_n: 每 N 块跑一次 FAISS（1 = 每块都跑）。
        blend_counter: 单元素列表 [count]，必须初始化为 [0]（首块即混合）。
        blend_cache: 单元素列表，缓存上次混合后的特征段。

    Returns:
        混合后的 feats（原地修改）。
    """
    if index is None or index_vectors is None or index_rate <= 0:
        return feats

    should_blend = True
    if blend_counter is not None and blend_every_n > 1:
        should_blend = blend_counter[0] == 0
        blend_counter[0] = (blend_counter[0] + 1) % blend_every_n

    start = skip_head // 2
    seg = feats[0][start:]

    if should_blend:
        try:
            npy = seg.detach().cpu().numpy().astype("float32")
            blended = faiss_blend(npy, index, index_vectors, index_rate, is_half)
            feats[0][start:] = torch.from_numpy(blended).to(device)
            if blend_cache is not None:
                blend_cache[0] = feats[0][start:].clone()
        except Exception as e:
            logger.warning("FAISS 索引混合失败，使用原始特征: %s", e)
    elif blend_cache is not None and blend_cache[0] is not None:
        cached = blend_cache[0]
        if cached.shape[0] == seg.shape[0]:
            feats[0][start:] = cached
    return feats
