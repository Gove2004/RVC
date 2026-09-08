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
    if use_f0 == 1 and protect > 0:
        return feats.clone()
    return None


def protect_blend(feats_converted: torch.Tensor, feats_original: torch.Tensor, pitchf: torch.Tensor, protect: float) -> torch.Tensor:
    # 浊音（pitchf 高）→ 全转换；清音（pitchf 低/0）→ 按 protect 混合原特征。
    # 软阈值（sigmoid）：在 pitchf 接近 0 的区域平滑过渡，减少浊音/清音边界突变，
    # 改善短辅音（b/p/d/t）被误判为浊音导致的咬字不清。
    from rvc.core.experimental import experimental_config
    if experimental_config.protect_soft_enabled:
        threshold = experimental_config.protect_soft_threshold_hz
        width = experimental_config.protect_soft_width
        # uv_prob: 0=浊音（全转换），1=UV/清音（按 protect 混合原特征）
        uv_prob = torch.sigmoid((threshold - pitchf) / (width / 4))
        mix = 1.0 - protect * uv_prob
    else:
        mix = torch.where(pitchf > 0, 1.0, 1.0 - protect)
    pitchff = mix.unsqueeze(-1)
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
