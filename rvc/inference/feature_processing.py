"""HuBERT 特征处理。"""
import torch
import torch.nn.functional as F


def extract_hubert_features(model, input_wav, device: str, is_half: bool) -> torch.Tensor:
    # 固定形状块不需要 padding mask：attention_mask=None（全 1）即为正确语义。
    # 历史上曾有 has_nonzero 标记机制，恒为 False 从未触发过 mask 分支，已删。
    if not torch.is_tensor(input_wav):
        input_wav = torch.from_numpy(input_wav)
    feats = input_wav.to(device)
    feats = feats.half() if is_half else feats.float()
    feats = feats.view(1, -1)

    # transformers 模型返回 BaseModelOutput（return_dict=True），取 last_hidden_state；
    # 兼容直接返回 tensor 的边缘模型。
    feats_result = model(feats).last_hidden_state
    # Unconditional last-frame padding for feature alignment
    feats_result = torch.cat((feats_result, feats_result[:, -1:, :]), 1)
    return feats_result


def clone_protect_source(feats: torch.Tensor, use_f0: int, protect: float) -> torch.Tensor | None:
    if use_f0 == 1 and protect > 0:
        return feats.clone()
    return None


def protect_blend(feats_converted: torch.Tensor, feats_original: torch.Tensor, pitchf: torch.Tensor, protect: float) -> torch.Tensor:
    # 浊音（pitchf > 0）→ 全转换；清音（pitchf = 0）→ 按 protect 混合原特征。
    # 用 torch.where 替代索引赋值，避免 [pitchf > 0] = 1 / [pitchf < 1] = 1-p 两行
    # 在 0 < f < 1 区间表面重叠造成的误解（实际 pitchf 是 Hz 值，不会落在此区间）。
    pitchff = torch.where(pitchf > 0, 1.0, 1.0 - protect).unsqueeze(-1)
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
