"""HuBERT 特征处理 — 特征提取 + 上采样。

简化后去掉了清辅音保护（protect_blend / clone_protect_source），
只保留核心的特征提取和上采样功能。
"""
import torch
import torch.nn.functional as F

from rvc.inference.cuda_graph import run_cuda_graph


def extract_hubert_features(model, input_wav, device: str, is_half: bool) -> torch.Tensor:
    """提取 HuBERT 特征，走 CUDA Graph 加速。

    固定形状块不需要 padding mask：attention_mask=None（全 1）即为正确语义。
    预处理（dtype/shape 转换）和后处理（末帧 padding）放在 CUDA Graph 外，
    只捕获模型前向传播（最耗时部分）。

    Args:
        model: HuBERT 模型
        input_wav: 16kHz 音频（1D tensor 或 numpy）
        device: 目标设备
        is_half: 是否使用 FP16

    Returns:
        HuBERT 特征 (1, T, 768)，50fps
    """
    if not torch.is_tensor(input_wav):
        input_wav = torch.from_numpy(input_wav)
    feats = input_wav.to(device)
    feats = feats.half() if is_half else feats.float()
    feats = feats.view(1, -1)

    def _hubert_forward(x):
        return model(x).last_hidden_state

    feats_result = run_cuda_graph(model, "hubert", _hubert_forward, feats)

    # 末帧 padding 用于特征对齐
    feats_result = torch.cat((feats_result, feats_result[:, -1:, :]), 1)
    return feats_result


def upsample_features(
    feats: torch.Tensor,
    p_len: int,
    is_half: bool,
) -> torch.Tensor:
    """特征上采样：50fps → 100fps（线性插值），截取 p_len 帧。

    Args:
        feats: HuBERT 特征 (B, T, C)，50fps
        p_len: 目标长度（10ms 帧数）
        is_half: 是否输出 half precision

    Returns:
        上采样后的特征 (B, p_len, C)，100fps
    """
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :p_len, :]
    if is_half:
        feats = feats.half()
    return feats
