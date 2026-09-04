"""特征索引加载与混合（torch-GPU 暴力近邻版，零 CPU 依赖）。

原实现基于 faiss-CPU（index.search + 每块 GPU→CPU→GPU 搬运），
实时链路实测一块检索 441ms、CPU 占用过高，已整体替换为：
  - 索引文件 = 归一化特征张量 .pt（免 faiss IVF 训练，加载即用）
  - 检索 = GPU 上 cos-sim matmul + topk，全程无 CPU 往返
    （5 万帧特征库实测 ~31ms/块，且 pipeline 已做 blend_every_n=4 降频）
"""
import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 检索近邻数（与上游 faiss 版一致）
_TOP_K = 8
# 余弦距离 1-sim 的下限，防止最近邻权重发散
_DIST_FLOOR = 1e-4
# 检索分块帧数：sim = (块帧 × N) 中间矩阵按块算，避免整段×全库一次性
# 撑爆显存（离线转换整段特征可达数万帧 × 几十万帧索引）。
_BLEND_CHUNK = 512


def load_index(index_path: str, inference_cache):
    """加载特征索引（.pt 张量，L2 归一化后缓存）。

    Returns:
        (refs, None): refs 为 (N, D) CPU tensor（fp16 或 fp32）；文件缺失/为空时返回 (None, None)。
        第二项保留 None 仅为兼容旧调用签名。
    """
    if not index_path or not os.path.exists(index_path):
        return None, None

    cached = inference_cache.get_index(index_path)
    if cached:
        logger.info("加载特征索引 %s（缓存）", os.path.basename(index_path))
        return cached["index"], None

    try:
        refs = torch.load(index_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("特征索引加载失败（忽略）: %s", index_path)
        return None, None

    if refs is None or refs.ndim != 2 or refs.shape[0] == 0:
        logger.warning("特征索引为空或形状异常: %s", index_path)
        return None, None

    refs = refs.detach()
    # 幂等 L2 归一化：构建端已归一化，这里兜底再归一化一次无副作用。
    # fp16 索引（index_builder 低内存版产物）保持 fp16 不升 fp32——
    # 检索精度实测无损（top1 与 fp32 一致率 100%），GPU 显存减半。
    if refs.dtype == torch.float16:
        refs = F.normalize(refs.float(), dim=1).half()
    else:
        refs = F.normalize(refs.float(), dim=1)
    inference_cache.set_index(index_path, {"index": refs, "index_vectors": None})
    logger.info("加载特征索引 %s（%d 帧 × %d 维）", os.path.basename(index_path), refs.shape[0], refs.shape[1])
    return refs, None


def _torch_blend(seg: torch.Tensor, refs: torch.Tensor, index_rate: float, is_half: bool) -> torch.Tensor:
    """GPU 暴力近邻混合：cos-sim top-k，按余弦距离反比加权，再按 index_rate 融合。

    seg  : (T, D) 设备上特征段（可能 fp16）
    refs : (N, D) L2 归一化特征库（fp16 或 fp32 均可，matmul 自动提升）
    返回 : 与 seg 同 dtype 的 (T, D)
    """
    orig_dtype = seg.dtype
    q = F.normalize(seg.float(), dim=1)          # (T, D)，仅用于相似度
    refs_d = refs.to(device=seg.device)          # 同卡时 no-op
    if refs_d.dtype == torch.float16:
        q = q.half()  # matmul 要求同 dtype；fp16 索引走 fp16 matmul（精度实测无损）
    k = min(_TOP_K, refs_d.shape[0])

    # 分块检索：sim 中间矩阵只保留 (块帧, N)，整段 T 很大时避免一次性
    # 分配 (T, N)（离线整段特征 × 几十万帧索引可达数十 GB）。
    pieces = []
    for qi in q.split(_BLEND_CHUNK, dim=0):
        sim = qi @ refs_d.t()                    # (块, N) cos-sim
        sim_top, ix = sim.topk(k, dim=1)         # 越近 → sim 越大
        if not torch.isfinite(sim_top).all():
            return seg
        # 距离反比权重：dist = 1 - sim，近邻 dist→0 权重发散，clamp 保底
        w = torch.reciprocal((1.0 - sim_top).clamp(min=_DIST_FLOOR))
        w = w / w.sum(dim=1, keepdim=True)
        pieces.append((refs_d[ix] * w.unsqueeze(-1)).sum(dim=1))  # (块, D)
    blended = torch.cat(pieces, dim=0)           # (T, D)

    result = blended * index_rate + seg.float() * (1.0 - index_rate)
    return result.to(orig_dtype)


def apply_faiss_index(
    feats: torch.Tensor,
    index,
    index_vectors,
    index_rate: float,
    is_half: bool,
    device: str,
    skip_head: int = 0,
    blend_every_n: int = 1,
    blend_counter: list[int] | None = None,
    blend_cache: list | None = None,
) -> torch.Tensor:
    """按 index_rate 混合特征（torch-GPU 版，签名与旧 faiss 版一致）。

    降频语义：每 blend_every_n 块真正跑一次检索；中间块沿用上一次的混合
    结果（而非原生特征），避免音色每 N 块周期性跳动，也摊薄检索开销。

    index: 由 load_index 返回的 (N, D) 归一化特征张量（fp16/fp32，CPU 或 GPU 均可）。
    """
    if index is None or index_rate <= 0:
        return feats

    should_blend = True
    if blend_counter is not None and blend_every_n > 1:
        should_blend = blend_counter[0] == 0
        blend_counter[0] = (blend_counter[0] + 1) % blend_every_n

    start = skip_head // 2
    seg = feats[0][start:]

    if should_blend:
        try:
            blended = _torch_blend(seg, index, index_rate, is_half)
            feats[0][start:] = blended
            if blend_cache is not None:
                blend_cache[0] = feats[0][start:].clone()
        except Exception as e:
            logger.warning("索引混合失败，使用原始特征: %s", e)
    elif blend_cache is not None and blend_cache[0] is not None:
        cached = blend_cache[0]
        if cached.shape[0] == seg.shape[0]:
            feats[0][start:] = cached
    return feats
