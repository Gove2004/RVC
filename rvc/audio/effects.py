"""音频效果链 — 机架风格的模块化处理单元

每个效果器都是独立的处理单元，有明确的输入输出接口。
效果链按顺序串联处理，支持实时和离线两种模式。
"""
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional

# 混响延迟配置（毫秒）
# 使用不同延迟时间创造空间感，避免梳状滤波
REVERB_DELAYS_MS = [17, 31, 47, 73]
REVERB_GAINS = [0.3, -0.2, 0.15, -0.08]  # 对应每个延迟的增益
REVERB_BUFFER_SIZE_MS = 150  # 实时模式缓冲区大小（毫秒）


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


class ParametricEQ(AudioEffect):
    """5段参数均衡器 — FFT频域处理（无状态，适合实时）

    频段：
    - 超低频 60Hz
    - 低频 200Hz
    - 中频 1kHz
    - 中高频 3kHz
    - 高频 8kHz
    """

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.bands = {
            'sub': {'center': 60, 'width': 40, 'gain_db': 0.0},      # 超低频
            'low': {'center': 200, 'width': 100, 'gain_db': 0.0},    # 低频
            'mid': {'center': 1000, 'width': 300, 'gain_db': 0.0},   # 中频
            'hi_mid': {'center': 3000, 'width': 600, 'gain_db': 0.0}, # 中高频
            'high': {'center': 8000, 'width': 2000, 'gain_db': 0.0}, # 高频
        }
        self._freqs_cache_key = None
        self._freqs_cache = None
        self._mask_cache = {}
        self._gain_curve_cache_key = None
        self._gain_curve_cache = None

    def set_band(self, band: str, gain_db: float):
        """设置频段增益

        Args:
            band: 'sub' | 'low' | 'mid' | 'hi_mid' | 'high'
            gain_db: 增益 (dB)，范围 -12 ~ +12
        """
        if band in self.bands:
            self.bands[band]['gain_db'] = max(-12, min(12, gain_db))

    def _get_freqs(self, audio: torch.Tensor) -> tuple[tuple, torch.Tensor]:
        key = (audio.shape[0], audio.device, audio.dtype)
        if self._freqs_cache_key != key:
            self._freqs_cache_key = key
            self._freqs_cache = torch.fft.rfftfreq(audio.shape[0], 1 / self.sample_rate, device=audio.device).to(audio.dtype)
            self._mask_cache.clear()
            self._gain_curve_cache_key = None
            self._gain_curve_cache = None
        return key, self._freqs_cache

    def _get_mask(self, band: str, freqs_key: tuple, freqs: torch.Tensor) -> torch.Tensor:
        key = (band, freqs_key)
        mask = self._mask_cache.get(key)
        if mask is None:
            band_cfg = self.bands[band]
            mask = torch.exp(-0.5 * ((freqs - band_cfg['center']) / band_cfg['width']) ** 2)
            self._mask_cache[key] = mask
        return mask

    def _get_gain_curve(self, audio: torch.Tensor) -> torch.Tensor:
        freqs_key, freqs = self._get_freqs(audio)
        gains = tuple(self.bands[band]['gain_db'] for band in self.bands)
        key = (freqs_key, gains)
        if self._gain_curve_cache_key == key and self._gain_curve_cache is not None:
            return self._gain_curve_cache

        gain_curve = torch.ones(freqs.shape, device=audio.device, dtype=audio.dtype)
        for band, band_cfg in self.bands.items():
            gain_db = band_cfg['gain_db']
            if gain_db == 0:
                continue
            gain_linear = 10 ** (gain_db / 20)
            gain_curve *= 1 + (gain_linear - 1) * self._get_mask(band, freqs_key, freqs)

        self._gain_curve_cache_key = key
        self._gain_curve_cache = gain_curve
        return gain_curve

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """FFT频域均衡处理"""
        # 快速路径：所有增益为 0 时跳过
        if all(b['gain_db'] == 0 for b in self.bands.values()):
            return audio

        spec = torch.fft.rfft(audio)
        spec *= self._get_gain_curve(audio)
        return torch.fft.irfft(spec, n=audio.shape[0])


class SimpleReverb(AudioEffect):
    """简单混响 — 多延迟叠加（Schroeder 风格）

    实时模式：维护延迟缓冲区（有状态）
    离线模式：全局卷积（无状态）
    """

    def __init__(self, sample_rate: int, realtime: bool = True):
        super().__init__(sample_rate)
        self.realtime = realtime
        self.mix = 0.0  # 混响混合比例 (0 = 干声, 1 = 湿声)

        # 延迟参数（毫秒）
        self.delays_ms = REVERB_DELAYS_MS
        self.delays_samples = [int(d * 0.001 * sample_rate) for d in self.delays_ms]
        self.gains = REVERB_GAINS

        # 实时模式：维护环形缓冲区（设备在首次处理时推断）
        if realtime:
            max_delay = max(self.delays_samples)
            buffer_size = int(REVERB_BUFFER_SIZE_MS * 0.001 * sample_rate)
            self.buffer = None  # 延迟初始化，等待推断 device
            self.buffer_size = buffer_size

    def set_mix(self, mix: float):
        """设置混响混合比例

        Args:
            mix: 0.0 ~ 0.5，超过 0.5 会过湿
        """
        self.mix = max(0.0, min(0.5, mix))

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """混响处理"""
        if self.mix == 0:
            return audio

        if self.realtime:
            return self._process_realtime(audio)
        else:
            return self._process_offline(audio)

    def _process_realtime(self, audio: torch.Tensor) -> torch.Tensor:
        """实时模式 — 使用环形缓冲区"""
        # 延迟初始化 buffer（推断 device）
        if self.buffer is None:
            self.buffer = torch.zeros(self.buffer_size, device=audio.device)

        # 拼接历史缓冲 + 当前音频
        full_audio = torch.cat([self.buffer, audio])

        # 计算混响尾音
        reverb = torch.zeros_like(audio)
        for delay, gain in zip(self.delays_samples, self.gains):
            start_idx = self.buffer_size - delay
            reverb += full_audio[start_idx : start_idx + audio.shape[0]] * gain

        # 平滑（3点移动平均）
        reverb = F.avg_pool1d(reverb[None, None, :], 5, 1, 2).squeeze()

        # 混合
        output = audio * (1 - self.mix * 0.5) + reverb * self.mix

        # 更新缓冲区
        self.buffer = full_audio[-self.buffer_size:]

        return output

    def _process_offline(self, audio: torch.Tensor) -> torch.Tensor:
        """离线模式 — 全局卷积（与实时模式一致：不延长尾音）"""
        reverb = torch.zeros_like(audio)

        # 应用各延迟通道
        for delay, gain in zip(self.delays_samples, self.gains):
            if delay < audio.shape[0]:
                # 延迟叠加（不超出原始长度）
                reverb[delay:] += audio[:-delay] * gain

        # 平滑混响（使用 pad 避免边界问题）
        reverb_padded = F.pad(reverb.unsqueeze(0).unsqueeze(0), (2, 2), mode='reflect')
        reverb = F.avg_pool1d(reverb_padded, 5, 1).squeeze()

        # 混合
        return audio * (1 - self.mix * 0.5) + reverb * self.mix

    def reset(self):
        """重置缓冲区"""
        if self.realtime and hasattr(self, 'buffer') and self.buffer is not None:
            self.buffer.zero_()


class EffectChain:
    """效果链 — 按顺序串联多个效果器"""

    def __init__(self):
        self.effects: list[AudioEffect] = []

    def add(self, effect: AudioEffect):
        """添加效果器到链尾"""
        self.effects.append(effect)
        return self

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """串联处理音频"""
        for effect in self.effects:
            audio = effect(audio)
        return audio

    def reset_all(self):
        """重置所有效果器"""
        for effect in self.effects:
            effect.reset()

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.process(audio)


def create_realtime_chain(sample_rate: int) -> tuple[EffectChain, ParametricEQ, SimpleReverb]:
    """创建实时效果链（用于音频回调）

    Returns:
        (chain, eq, reverb) — 链对象 + EQ引用 + 混响引用（便于外部调参）
    """
    eq = ParametricEQ(sample_rate)
    reverb = SimpleReverb(sample_rate, realtime=True)

    chain = EffectChain()
    chain.add(eq)
    chain.add(reverb)

    return chain, eq, reverb


def create_offline_chain(sample_rate: int) -> tuple[EffectChain, ParametricEQ, SimpleReverb]:
    """创建离线效果链（用于文件转换）

    Returns:
        (chain, eq, reverb) — 链对象 + EQ引用 + 混响引用
    """
    eq = ParametricEQ(sample_rate)
    reverb = SimpleReverb(sample_rate, realtime=False)

    chain = EffectChain()
    chain.add(eq)
    chain.add(reverb)

    return chain, eq, reverb
