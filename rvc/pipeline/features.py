"""HuBERT 特征处理 — 特征提取 + 上采样。

特征侧不做任何保护性 padding：末帧 padding 统一在 upsample_features 中做。
"""
import torch
import torch.nn.functional as F

from rvc.runtime.graph import run_cuda_graph

# 推理时在 16k 窗口右端补的静音长度（0.5s）。HuBERT 是非因果模型，同一帧
# 的特征取决于它右边的未来上下文；滑窗消费区贴着窗口右边缘，同一帧内容
# 在不同块里「未来帧数」不同，特征随之跳变，持续/重复发声时输出在两个
# 渲染吸引态之间翻转（字漂移）。补足右端固定静音后，每帧特征只依赖
# <=t 的音频，逐块一致。代价：消费帧的特征带「未来是静音」偏置（训练侧
# 提取时未来是真实语音的分布偏移），实测对正常语音无损、对持续音大幅
# 变稳；训练侧 hubert_forward 不补零，两边语义各自成立。
HUBERT_RIGHT_PAD = 8000
_HUBERT_PAD_FRAMES = HUBERT_RIGHT_PAD // 320   # 补零折合的特征帧数


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
    预处理（dtype/shape 转换、右端补零）放在 CUDA Graph 外，只捕获模型前向传播。
    末帧 padding 已移到 upsample_features 中做，本函数只负责提取。

    右端补 HUBERT_RIGHT_PAD 静音再提取、再切掉补零折合的帧数：输出形状与
    不补零时完全一致（同一帧始终是同一帧），但消费区特征因果化、逐块一致
    ——机制见 HUBERT_RIGHT_PAD 注释。

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
    pad = torch.zeros(1, HUBERT_RIGHT_PAD, dtype=feats.dtype, device=feats.device)
    feats = torch.cat([feats, pad], dim=1)

    # hubert_forward 是共享函数（训练侧直接调用），推理侧用闭包绑定 model 以适配 CUDA Graph
    def _forward(x):
        return hubert_forward(model, x)

    out = run_cuda_graph(model, "hubert", _forward, feats)
    return out[:, :out.shape[1] - _HUBERT_PAD_FRAMES, :]


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
