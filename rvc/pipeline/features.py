"""HuBERT 特征处理 — 特征提取 + 上采样。

在特征侧不再做保护。
"""
import torch
import torch.nn.functional as F

from rvc.runtime.graph import run_cuda_graph


def extract_hubert_features(model, input_wav, device: str, is_half: bool) -> torch.Tensor:
    """提取 HuBERT 特征，走 CUDA Graph 加速。

    固定形状块不需要 padding mask：attention_mask=None（全 1）即为正确语义。
    预处理（dtype/shape 转换）放在 CUDA Graph 外，只捕获模型前向传播。
    末帧 padding 已移到 upsample_features 中做，本函数只负责提取。

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

    return run_cuda_graph(model, "hubert", _hubert_forward, feats)


def upsample_features(
    feats: torch.Tensor,
    p_len: int,
    is_half: bool,
) -> torch.Tensor:
    """特征上采样：50fps → 100fps（最近邻插值，与上游一致），截取 p_len 帧。

    末帧 padding（重复最后一帧）在此处做，确保上采样后长度足够覆盖 p_len。

    Args:
        feats: HuBERT 特征 (B, T, C)，50fps
        p_len: 目标长度（10ms 帧数）
        is_half: 是否输出 half precision

    Returns:
        上采样后的特征 (B, p_len, C)，100fps
    """
    # 末帧 padding：重复最后一帧，确保上采样后长度足够
    feats = torch.cat((feats, feats[:, -1:, :]), dim=1)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    # 长度不足 p_len 时必须显式失败：静默截短会让下游合成器的
    # 特征/F0 掩码长度错位，产生错音或崩溃，且极难排查
    if feats.shape[1] < p_len:
        raise ValueError(
            f"upsample_features: 上采样后长度 {feats.shape[1]} < 目标 {p_len}"
            f"（输入 50fps 帧数 {feats.shape[1] // 2} 不足）"
        )
    feats = feats[:, :p_len, :]

    feats = feats.half() if is_half else feats.float()
    return feats
