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
        return cached["index"], cached["index_vectors"]

    index = faiss.read_index(index_path)
    index_vectors = index.reconstruct_n(0, index.ntotal)
    inference_cache.set_index(index_path, {
        "index": index,
        "index_vectors": index_vectors,
    })
    logger.info("加载 Index: %s", os.path.basename(index_path))
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
    blend_every_n: int = 4,
    blend_counter: list[int] | None = None,
) -> tuple[torch.Tensor, int | None]:
    """Apply FAISS index blending with optional frequency reduction.

    Args:
        blend_every_n: Run FAISS every Nth block (1 = every block). Reduces GPU-CPU-GPU sync overhead.
        blend_counter: Mutable single-element list [count]. If None, FAISS runs every block.

    Returns:
        (feats, blend_counter) — counter is None if blend_every_n <= 1 or no counter provided.
    """
    if index is None or index_vectors is None or index_rate <= 0:
        return feats, blend_counter

    should_blend = True
    if blend_counter is not None and blend_every_n > 1:
        blend_counter[0] = (blend_counter[0] + 1) % blend_every_n
        should_blend = blend_counter[0] == 0

    if not should_blend:
        return feats, blend_counter

    try:
        npy = feats[0][skip_head // 2:].detach().cpu().numpy().astype("float32")
        blended = faiss_blend(npy, index, index_vectors, index_rate, is_half)
        feats[0][skip_head // 2:] = torch.from_numpy(blended).to(device)
    except Exception as e:
        logger.warning("FAISS 索引混合失败，使用原始特征: %s", e)
    return feats, blend_counter
