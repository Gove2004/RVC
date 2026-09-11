"""HuBERT 特征处理。"""
import torch
import torch.nn.functional as F

from rvc.inference.cuda_graph import run_cuda_graph

def extract_hubert_features(model, input_wav, device: str, is_half: bool) -> torch.Tensor:
    """提取 HuBERT 特征，走 CUDA Graph 加速。

    固定形状块不需要 padding mask：attention_mask=None（全 1）即为正确语义。
    预处理（dtype/shape 转换）和后处理（末帧 padding）放在 CUDA Graph 外，
    只捕获模型前向传播（最耗时部分）。
    """
    if not torch.is_tensor(input_wav):
        input_wav = torch.from_numpy(input_wav)
    feats = input_wav.to(device)
    feats = feats.half() if is_half else feats.float()
    feats = feats.view(1, -1)

    def _hubert_forward(x):
        return model(x).last_hidden_state

    feats_result = run_cuda_graph(model, "hubert", _hubert_forward, feats)

    # Unconditional last-frame padding for feature alignment
    feats_result = torch.cat((feats_result, feats_result[:, -1:, :]), 1)
    return feats_result

def clone_protect_source(feats: torch.Tensor, use_f0: int, protect: float) -> torch.Tensor | None:
    """返回需要保护的原始特征引用。

    注意：不再克隆，因为后续 F.interpolate 会创建新张量，不会修改输入。
    这样可以避免一次不必要的张量复制（768 维 × 数百帧）。
    """
    if use_f0 == 1 and protect > 0:
        return feats
    return None

def protect_blend(
    feats_converted: torch.Tensor,
    feats_original: torch.Tensor,
    protect: float,
    uv_prob: torch.Tensor,
) -> torch.Tensor:
    """辅音保护特征混合：浊音全转换，清音按 protect 混合原特征。

    Args:
        feats_converted: 转换后的 HuBERT 特征 (B, T, C)
        feats_original: 原始 HuBERT 特征 (B, T, C)
        protect: 辅音保护强度 0~1（0=全转换，1=清音全保留）
        uv_prob: 清浊概率 (B, T)，0=浊音，1=清音/UV，
                 由 voicing.compute_uv_prob 多特征融合计算

    Returns:
        混合后的特征 (B, T, C)
    """
    mix = 1.0 - protect * uv_prob
    pitchff = mix.unsqueeze(-1)
    return feats_converted * pitchff + feats_original * (1 - pitchff)

def upsample_features(
    feats: torch.Tensor,
    p_len: int,
    is_half: bool,
    feats0: torch.Tensor | None = None,
    protect: float = 0.0,
    uv_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """特征上采样 + 辅音保护混合。

    Args:
        feats: HuBERT 特征 (B, T, C)
        p_len: 目标长度（10ms 帧数）
        is_half: 是否输出 half precision
        feats0: 原始特征克隆（用于辅音保护混合），None 时不混合
        protect: 辅音保护强度 0~1
        uv_prob: 清浊概率 (B, T)，必须与 feats0 同时提供才进行混合
    """
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :p_len, :]
    if feats0 is not None and uv_prob is not None:
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats0 = feats0[:, :p_len, :]
        feats = protect_blend(feats, feats0, protect, uv_prob)
        if is_half:
            feats = feats.half()
    return feats
