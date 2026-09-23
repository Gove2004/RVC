"""实时 F0 提取 — 每块直接从完整 input_wav 提取，与 HuBERT 输入对齐。

设计思路：
- 以前用滚动缓存：每块只提取新增部分，写入缓存尾部
- 问题：F0 提取窗口和 HuBERT 输入窗口长度不一样，导致帧时间轴偏移
- 现在：每块直接从完整 input_wav_16k 提取 F0，和 HuBERT 输入完全一致
- 收益：帧天然对齐，消除快速声调转折处的咬字变怪问题
- 代价：F0 计算量增加（RMVPE 本来就快，延迟增加 <1ms）
"""
from rvc.core.constants import HUBERT_SAMPLE_RATE
import torch

from rvc.pipeline.pitch.extractor import create_f0_extractor


def extract_f0_for_block(
    input_wav: torch.Tensor,
    p_len: int,
    method: str,
    device: str,
    is_half: bool,
    inference_cache,
    config=None,
    extractor=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从完整 input_wav 提取 F0，返回最后 p_len 帧。

    与 HuBERT 输入完全一致（都是完整 input_wav_16k），帧天然对齐。

    注意：不再丢弃边缘帧！以前丢弃是因为滚动缓存会把边缘帧滚到中间，
    现在每块都从完整缓冲区重新提取，边缘帧就是头部旧音频，
    我们取最后 p_len 帧时自然就避开了边缘帧。
    而且最后 p_len 帧对应的是最近的音频，relevance 足够，不存在边缘不可信的问题。

    Args:
        input_wav: 16k 滚动缓冲区（完整长度，和 HuBERT 输入一致）
        p_len: 需要返回的 F0 帧数（100fps）
        method: F0 提取方法（rmvpe/fcpe）
        device/is_half/inference_cache: 设备和缓存
        config: InferenceParams
        extractor: 已缓存的 F0 提取器实例（None 时内部创建）

    Returns:
        (f0_raw, confidence_raw): 原始连续 F0 (p_len,)、原始 confidence (p_len,)
    """
    if extractor is None:
        extractor = create_f0_extractor(method, device, is_half, inference_cache, config=config)

    # 从完整 input_wav 提取 F0，和 HuBERT 输入完全一致
    f0_raw, confidence_raw = extractor.extract_raw(
        input_wav, HUBERT_SAMPLE_RATE,
    )

    # 对齐到 p_len 帧：和 HuBERT 特征的前 p_len 帧对齐
    # 特征是 upsample 后取前 p_len 帧，所以 F0 也应该取前 p_len 帧
    n = f0_raw.shape[0]
    if n >= p_len:
        # 帧数足够，取前 p_len 帧（和特征对齐）
        f0 = f0_raw[:p_len]
        conf = confidence_raw[:p_len]
    else:
        # 帧数不足，尾部补零
        pad_len = p_len - n
        f0 = torch.nn.functional.pad(f0_raw, (0, pad_len))
        conf = torch.nn.functional.pad(confidence_raw, (0, pad_len))

    return f0, conf
