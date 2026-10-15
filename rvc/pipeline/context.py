"""单块推理上下文 — 每块复用同一实例，调用 reset() 重置。

所有只在当前块内有效的中间产物都放这里。
每个阶段填充自己的字段，下游阶段读取。

设计原则：
- 复用同一实例，避免每块分配 dataclass 的开销
- reset() 直接赋值字段，避免 dataclasses.fields 反射开销（热路径每块调用）
- 字段按数据流顺序排列
"""
import torch

from rvc.core.config import InferenceParams


class InferenceContext:
    """单块推理上下文 — 每块 reset() 后复用。

    不用 @dataclass：热路径每块调用 reset()，dataclasses.fields 反射比直接赋值慢一个数量级。
    字段少，直接手写 reset 既快又清晰。
    """

    __slots__ = (
        "config",
        "p_len",
        "f0_raw",
        "_f0_raw_untruncated",
        "formant_factor",
        "return_length2",
    )

    def __init__(self):
        self.config: InferenceParams | None = None
        self.p_len: int = 0
        self.f0_raw: torch.Tensor | None = None
        self._f0_raw_untruncated: torch.Tensor | None = None
        self.formant_factor: float = 1.0
        self.return_length2: int = 0

    def reset(self, config: InferenceParams) -> None:
        """每块开始时重置所有字段，复用实例避免分配开销。"""
        self.config = config
        self.p_len = 0
        self.f0_raw = None
        self._f0_raw_untruncated = None
        self.formant_factor = 1.0
        self.return_length2 = 0
