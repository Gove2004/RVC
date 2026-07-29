"""HuBERT 特征处理。"""
import torch
import torch.nn.functional as F


def cached_padding_mask(cache: dict, shape, device: str) -> tuple[torch.Tensor, bool]:
    entry = cache.get(shape)
    if entry is None:
        if len(cache) >= 8:
            cache.clear()
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        cache[shape] = (mask, False)  # (mask, has_nonzero)
        return mask, False
    mask, has_nonzero = entry
    return mask, has_nonzero


def extract_hubert_features(model, input_wav, device: str, is_half: bool, padding_mask_cache: dict) -> torch.Tensor:
    if not torch.is_tensor(input_wav):
        input_wav = torch.from_numpy(input_wav)
    feats = input_wav.to(device)
    feats = feats.half() if is_half else feats.float()
    feats = feats.view(1, -1)
    padding_mask, _has_padding = cached_padding_mask(padding_mask_cache, feats.shape, device)

    # For fixed-size realtime blocks the mask is pre-allocated zeros;
    # _has_padding tracks changes without a GPU→CPU sync.
    if not _has_padding:
        feats_result = model(feats, attention_mask=None)
    else:
        attention_mask = (~padding_mask.bool()).long()
        feats_result = model(feats, attention_mask=attention_mask)

    # CUDA Graph replay 返回的已经是 clone，但边缘情况可能不走图
    # Unpack tuple return (some models return (hidden,) not plain tensor)
    if isinstance(feats_result, tuple):
        feats_result = feats_result[0]
    # Unconditional last-frame padding for feature alignment
    feats_result = torch.cat((feats_result, feats_result[:, -1:, :]), 1)
    return feats_result


def clone_protect_source(feats: torch.Tensor, use_f0: int, protect: float) -> torch.Tensor | None:
    if use_f0 == 1 and protect > 0:
        return feats.clone()
    return None


def protect_blend(feats_converted: torch.Tensor, feats_original: torch.Tensor, pitchf: torch.Tensor, protect: float) -> torch.Tensor:
    pitchff = pitchf.clone()
    pitchff[pitchf > 0] = 1
    pitchff[pitchf < 1] = 1 - protect
    pitchff = pitchff.unsqueeze(-1)
    return feats_converted * pitchff + feats_original * (1 - pitchff)


def upsample_features(
    feats: torch.Tensor,
    p_len: int,
    is_half: bool,
    feats0: torch.Tensor | None = None,
    pitchf: torch.Tensor | None = None,
    protect: float = 0.0,
) -> torch.Tensor:
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :p_len, :]
    if feats0 is not None and pitchf is not None:
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats0 = feats0[:, :p_len, :]
        feats = protect_blend(feats, feats0, pitchf, protect)
        if is_half:
            feats = feats.half()
    return feats
