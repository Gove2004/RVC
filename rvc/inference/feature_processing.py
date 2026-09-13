"""HuBERT 特征处理 — 特征提取 + 上采样 + 清辅音保护。

清辅音保护：对 F0=0 的清音帧混合原始输入特征，避免合成器对无 F0 帧过度处理。
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
    feats0: torch.Tensor | None = None,
    pitchf: torch.Tensor | None = None,
    protect: float = 0.5,
) -> torch.Tensor:
    """特征上采样：50fps → 100fps（线性插值），截取 p_len 帧。

    清辅音保护：对 F0=0 的清音帧混合原始输入特征（feats0），
    避免合成器对无 F0 的辅音帧过度处理产生电音/撕裂。

    Args:
        feats: HuBERT 特征 (B, T, C)，50fps
        p_len: 目标长度（10ms 帧数）
        is_half: 是否输出 half precision
        feats0: 原始输入特征 (B, T, C)，用于清音保护；None 时不保护
        pitchf: 连续 F0 (B, T)，F0=0 的帧视为清音；None 时不保护
        protect: 保护强度（0-0.5，0.5=关闭，越小保护越强）

    Returns:
        上采样后的特征 (B, p_len, C)，100fps
    """
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :p_len, :]

    # 清辅音保护：F0=0 的清音帧混合原始特征
    if feats0 is not None and pitchf is not None and protect < 0.5:
        strength = 1.0 - protect / 0.5  # protect=0.33 → strength=0.34
        feats0_up = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats0_up = feats0_up[:, :p_len, :]
        # pitchf 长度对齐到 p_len（不足补 0 视为清音，超出截断）
        pitchf_aligned = torch.zeros(pitchf.shape[0], p_len, device=pitchf.device, dtype=pitchf.dtype)
        pitchf_aligned[:, :min(p_len, pitchf.shape[1])] = pitchf[:, :min(p_len, pitchf.shape[1])]
        uv_mask = (pitchf_aligned == 0).float()[:, :, None]  # [B, T, 1]
        # 清音帧在合成特征和原始特征之间混合，浊音帧保持合成特征不变
        feats = feats + strength * uv_mask * (feats0_up - feats)

    if is_half:
        feats = feats.half()
    return feats
