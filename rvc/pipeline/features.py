"""HuBERT 特征处理 — 特征提取 + 上采样。

特征侧不做任何保护性 padding：末帧 padding 统一在 upsample_features 中做。
"""
import torch
import torch.nn.functional as F

from rvc.runtime.graph import run_cuda_graph


def hubert_forward(model, x: torch.Tensor) -> torch.Tensor:
    """HuBERT 模型前向传播 — 训练侧和推理侧共享。

    Args:
        model: HuBERT 模型
        x: 输入音频 (1, T)，已在正确设备和 dtype 上

    Returns:
        last_hidden_state (1, T', 768)
    """
    return model(x).last_hidden_state


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

    # hubert_forward 是共享函数（训练侧直接调用），推理侧用闭包绑定 model 以适配 CUDA Graph
    def _forward(x):
        return hubert_forward(model, x)

    return run_cuda_graph(model, "hubert", _forward, feats)


def hubert_frame_count(hubert_model, n_samples: int) -> int:
    """HuBERT 对长 n_samples 的 16k 波形输出多少帧（50fps）。

    特征提取器是纯卷积栈，`padding=0`，逐层 `(L - k) // s + 1`。用它而不是
    「按 block_samples 比例估算」：后者与实际帧数差一帧会让预生成的 z 噪声场
    形状对不上 m_p，被形状校验静默拒绝，固定噪声场完全失效且不报错。

    Args:
        hubert_model: 已加载的 HuBERT（读 conv_kernel / conv_stride）
        n_samples: 16k 输入样本数

    Returns:
        输出帧数
    """
    cfg = hubert_model.config
    length = int(n_samples)
    for kernel, stride in zip(cfg.conv_kernel, cfg.conv_stride):
        length = (length - kernel) // stride + 1
    return length


def upsample_features(
    feats: torch.Tensor,
    is_half: bool,
) -> torch.Tensor:
    """特征上采样：50fps → 100fps（最近邻插值，与上游一致）。

    末帧 padding（重复最后一帧）在此处做。
    返回完整的 upsample 后特征，长度由调用方决定怎么用。

    Args:
        feats: HuBERT 特征 (B, T, C)，50fps
        is_half: 是否输出 half precision

    Returns:
        上采样后的特征 (B, T_100fps, C)，100fps
    """
    # 末帧 padding：重复最后一帧
    feats = torch.cat((feats, feats[:, -1:, :]), dim=1)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    feats = feats.half() if is_half else feats.float()
    return feats
