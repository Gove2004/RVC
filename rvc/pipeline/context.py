"""单块推理上下文 — 每块复用同一实例，调用 reset() 重置。

所有只在当前块内有效的中间产物都放这里。
每个阶段填充自己的字段，下游阶段读取。

设计原则：
- 复用同一实例，避免每块分配 dataclass 的开销
- reset() 重置所有字段为默认值
- 字段按数据流顺序排列
"""
from dataclasses import dataclass, fields

import torch

from rvc.core.config import InferenceParams


@dataclass
class InferenceContext:
    """单块推理上下文 — 每块 reset() 后复用。"""

    # ── 配置（每块传入）──
    config: InferenceParams | None = None

    # ── 特征上采样后由 postprocess_features 设置 ──
    p_len: int = 0

    # ── F0 原始提取（tracker 写入，后处理读取）──
    f0_raw: torch.Tensor | None = None
    confidence_raw: torch.Tensor | None = None

    # ── formant（extract_features 写入，formant 重采样/F0 缩放读取）──
    formant_factor: float = 1.0
    return_length2: int = 0

    def reset(self, config: InferenceParams) -> None:
        """每块开始时重置所有字段，复用实例避免分配开销。"""
        self.config = config
        for f in fields(self):
            if f.name != "config":
                setattr(self, f.name, f.default)
