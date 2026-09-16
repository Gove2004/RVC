"""单块推理上下文 — 每块复用同一实例，调用 reset() 重置。

所有只在当前块内有效的中间产物都放这里。
每个阶段填充自己的字段，下游阶段读取。

设计原则：
- 复用同一实例，避免每块分配 dataclass 的开销
- reset() 重置所有字段为默认值
- 字段按阶段分组，便于追踪数据流
"""
from dataclasses import dataclass, fields

import numpy as np
import torch

from rvc.core.config import InferenceParams


@dataclass
class InferenceContext:
    """单块推理上下文 — 每块 reset() 后复用。"""

    # ── 配置（每块传入，不可变）──
    config: InferenceParams | None = None
    block_frame_16k: int = 0
    block_frame_48k: int = 0

    # ── stage_input（硬件输入 + 输入预处理）──
    raw_input_np: np.ndarray | None = None
    raw_input_mono: torch.Tensor | None = None
    p_len: int = 0

    # ── stage_features（HuBERT 特征提取）──
    hubert_features: torch.Tensor | None = None

    # ── stage_f0（F0 原始提取；formant 因子也在本阶段算出供 stage_synthesis 复用）──
    f0_raw: torch.Tensor | None = None
    confidence_raw: torch.Tensor | None = None
    formant_factor: float = 1.0  # 2^(formant/12)，A4 正式字段（每块由 stage_features 必写）

    # ── stage_synthesis 预处理（F0 后处理 + 特征上采样 + 合成）──
    f0_mapped: torch.Tensor | None = None       # 音域映射后
    f0_filtered: torch.Tensor | None = None     # 中值滤波后
    pitch_discrete: torch.Tensor | None = None  # 离散 F0
    pitchf_continuous: torch.Tensor | None = None  # 连续 F0
    features_upsampled: torch.Tensor | None = None  # 上采样后特征 (100fps)
    synthesized_audio: torch.Tensor | None = None

    # ── stage_output（formant 重采样 + 效果器 + 硬件输出）──
    formanted_audio: torch.Tensor | None = None
    rms_matched_audio: torch.Tensor | None = None
    final_output: torch.Tensor | None = None
    output_np: np.ndarray | None = None

    def reset(self, config: InferenceParams, block_frame_16k: int, block_frame_48k: int) -> None:
        """每块开始时重置所有字段，复用实例避免分配开销。

        Args:
            config: 本块推理参数
            block_frame_16k: 本块新增的 16k 样本数
            block_frame_48k: 本块新增的 48k 样本数
        """
        self.config = config
        self.block_frame_16k = block_frame_16k
        self.block_frame_48k = block_frame_48k
        # 重置其余所有字段为默认值（从 dataclass field 读取）
        for f in fields(self):
            if f.name not in ("config", "block_frame_16k", "block_frame_48k"):
                setattr(self, f.name, f.default)
