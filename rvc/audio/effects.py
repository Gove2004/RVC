"""音频效果器抽象 — 统一降噪/RMS混合/SOLA时间对齐的接口。

历史上效果器调用散落在 RealtimeEngine._cb_impl 里，每个效果器有独立的
函数签名和调用时机。本模块提供 AudioEffect 抽象基类，所有效果器实现
统一的 process() 接口，便于效果器链的组合和测试。

注意：SOLA 是时间对齐算法，严格来说不是"效果器"，但为了统一接口
也纳入本模块。
"""
import logging
from abc import ABC, abstractmethod

import numpy as np

from rvc.audio.denoise import SpectralSubtraction

logger = logging.getLogger(__name__)


class AudioEffect(ABC):
    """音频效果器抽象基类。"""

    @abstractmethod
    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """处理音频块。

        Args:
            audio: 输入音频 (numpy float32, shape [samples] 或 [channels, samples])
            **kwargs: 效果器特定参数

        Returns:
            处理后的音频
        """


class DenoiseEffect(AudioEffect):
    """谱减法降噪效果器。"""

    def __init__(self, sr: int = 48000):
        self._denoiser = SpectralSubtraction(sr)
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def set_strength(self, strength: float):
        """设置降噪强度 [0, 1]。"""
        self._denoiser.set_strength(strength)

    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        if not self._enabled:
            return audio
        return self._denoiser.process(audio)


class RmsMixEffect(AudioEffect):
    """RMS 响度混合效果器 — 混合源音频和目标音频的响度。

    rms_mix=0 → 完全用目标响度，rms_mix=1 → 完全用源响度。
    """

    def __init__(self):
        self._rms_mix = 0.0

    @property
    def rms_mix(self) -> float:
        return self._rms_mix

    @rms_mix.setter
    def rms_mix(self, value: float):
        self._rms_mix = max(0.0, min(1.0, value))

    def process(self, audio: np.ndarray, source: np.ndarray = None, **kwargs) -> np.ndarray:
        """混合源音频和目标音频的 RMS 响度。

        Args:
            audio: 目标音频（变声后）
            source: 源音频（变声前），None 时直接返回 audio
        """
        if source is None or self._rms_mix == 0.0:
            return audio
        from rvc.audio.realtime_mix import apply_rms_mix
        return apply_rms_mix(audio, source, self._rms_mix)


class SolaEffect(AudioEffect):
    """SOLA 时间对齐效果器 — 消除块间不连续（相位对齐 + 交叉淡化）。

    注意：SOLA 需要额外的状态（sola_buffer, crossfade 窗口），
    所以它不是纯函数。每次 process() 会更新内部状态。
    """

    def __init__(self, crossfade_samples: int, sola_search_samples: int):
        self._crossfade_samples = crossfade_samples
        self._sola_search_samples = sola_search_samples
        self._sola_buffer = None  # 上一块的尾部，用于相位对齐

    def reset(self):
        """重置 SOLA 状态（切换流/文件时调用）。"""
        self._sola_buffer = None

    def process(self, audio: np.ndarray, prev_tail: np.ndarray = None, **kwargs) -> np.ndarray:
        """对音频块做 SOLA 时间对齐。

        Args:
            audio: 当前块音频
            prev_tail: 上一块的尾部（用于交叉淡化），None 时用内部 sola_buffer
        """
        from rvc.audio.sola import apply_sola
        if prev_tail is not None:
            self._sola_buffer = prev_tail
        if self._sola_buffer is None:
            # 第一块，无需对齐
            self._sola_buffer = audio[-self._crossfade_samples * 2:].copy()
            return audio
        aligned, new_tail = apply_sola(
            audio, self._sola_buffer,
            self._crossfade_samples, self._sola_search_samples,
        )
        self._sola_buffer = new_tail
        return aligned


class EffectChain:
    """效果器链 — 按顺序应用多个效果器。"""

    def __init__(self):
        self._effects: list[AudioEffect] = []

    def add(self, effect: AudioEffect):
        self._effects.append(effect)

    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        for effect in self._effects:
            audio = effect.process(audio, **kwargs)
        return audio

    def reset(self):
        """重置所有有状态的效果器。"""
        for effect in self._effects:
            effect.reset()
