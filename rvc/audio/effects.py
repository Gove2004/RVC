"""音频效果器基类 — 机架单元抽象接口（EQ/混响已移除，保留基类供降噪等使用）"""
import torch
from abc import ABC, abstractmethod


class AudioEffect(ABC):
    """音频效果器基类 — 机架单元抽象接口"""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.enabled = True

    @abstractmethod
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """处理音频

        Args:
            audio: 输入音频 (shape: [samples])

        Returns:
            处理后的音频 (shape: [samples])
        """
        pass

    def reset(self):
        """重置效果器状态（用于实时流切换场景）"""
        pass

    def set_enabled(self, enabled: bool):
        """启用/禁用效果器"""
        self.enabled = enabled

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """调用接口 — 支持 effect(audio) 语法"""
        if not self.enabled:
            return audio
        return self.process(audio)
