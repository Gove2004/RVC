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
        logger.info("使用缓存 Index: %s", os.path.basename(index_path))
        return cached["index"], cached["big_npy"]

    index = faiss.read_index(index_path)
    big_npy = index.reconstruct_n(0, index.ntotal)
    inference_cache.set_index(index_path, {
        "index": index,
        "big_npy": big_npy,
    })
    logger.info("加载 Index: %s", os.path.basename(index_path))
    return index, big_npy


def faiss_blend(feats_npy: np.ndarray, index, big_npy: np.ndarray, index_rate: float, is_half: bool) -> np.ndarray:
    k = min(8, index.ntotal)
    score, ix = index.search(feats_npy, k=k)
    if not (ix >= 0).all():
        return feats_npy
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    blended = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
    result = blended * index_rate + feats_npy * (1 - index_rate)
    if is_half:
        result = result.astype("float16")
    return result


def apply_faiss_index(
    feats: torch.Tensor,
    index,
    big_npy: np.ndarray | None,
    index_rate: float,
    is_half: bool,
    device: str,
    skip_head: int = 0,
) -> torch.Tensor:
    if index is None or big_npy is None or index_rate <= 0:
        return feats

    try:
        npy = feats[0][skip_head // 2:].detach().cpu().numpy().astype("float32")
        blended = faiss_blend(npy, index, big_npy, index_rate, is_half)
        feats[0][skip_head // 2:] = torch.from_numpy(blended).to(device)
    except Exception as e:
        logger.warning("FAISS 索引混合失败，使用原始特征: %s", e)
    return feats
