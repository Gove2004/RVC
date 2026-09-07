"""音频效果器 — 降噪 / RMS 混合 / SOLA 时间对齐，统一 torch.Tensor（GPU）接口。

RealtimeEngine._cb_impl 不再直接调用各底层函数，而是通过 AudioProcessor 编排：
  输入侧：process_input(mono, config) → 降噪
  输出侧：process_output(infer, ref, config) → RMS 混合 + SOLA

所有效果器在 setup() 时初始化（与 _init_processing 同时），process() 时零分配。
"""
import logging

import numpy as np
import torch

from rvc.audio.denoise import SpectralSubtraction
from rvc.audio.realtime_mix import apply_rms_mix
from rvc.audio.sola import apply_sola

logger = logging.getLogger(__name__)


class DenoiseEffect:
    """输入侧谱减法降噪。"""

    def __init__(self):
        self._nr_ss = None
        self._last_strength = None

    def setup(self, sr: int):
        self._nr_ss = SpectralSubtraction(sr)
        self._nr_ss.reset()
        self._last_strength = None

    def reset(self):
        if self._nr_ss is not None:
            self._nr_ss.reset()

    def process(self, mono: torch.Tensor, enable: bool, strength: float) -> torch.Tensor:
        if not enable or self._nr_ss is None:
            return mono
        if self._last_strength != strength:
            self._nr_ss.set_strength(strength)
            self._last_strength = strength
        return self._nr_ss(mono)


class RmsMixEffect:
    """RMS 响度包络混合 — 让转换后音量参考原始音量。"""

    def __init__(self):
        self._hz_centis = 0

    def setup(self, sr: int):
        self._hz_centis = sr // 100

    def process(self, infer: torch.Tensor, ref: torch.Tensor, rms_mix: float) -> torch.Tensor:
        if rms_mix >= 1.0:
            return infer
        return apply_rms_mix(ref, infer, rms_mix, self._hz_centis)


class SolaEffect:
    """SOLA 时间对齐 — 消除块间不连续（相位对齐 + 交叉淡化）。

    持有 sola_buffer / fade_in / fade_out / sola_norm_kernel 等预分配状态，
    setup() 时一次性分配，process() 时零分配、原地更新 sola_buffer。
    """

    def __init__(self):
        self.sola_buffer = None
        self._fade_in = None
        self._fade_out = None
        self._sola_norm_kernel = None
        self._block_samples = 0
        self._sola_buffer_samples = 0
        self._sola_search_samples = 0

    def setup(self, sr: int, block_samples: int, crossfade_samples: int,
              sola_search_samples: int, device: str):
        zc = sr // 100
        self._block_samples = block_samples
        self._sola_buffer_samples = min(crossfade_samples, 4 * zc)
        self._sola_search_samples = sola_search_samples
        self.sola_buffer = torch.zeros(self._sola_buffer_samples, device=device)
        ls = torch.linspace(0, 1, steps=self._sola_buffer_samples, device=device)
        self._fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self._fade_out = 1 - self._fade_in
        self._sola_norm_kernel = torch.ones(1, 1, self._sola_buffer_samples, device=device)

    def reset(self):
        if self.sola_buffer is not None:
            self.sola_buffer.zero_()

    def process(self, infer: torch.Tensor) -> torch.Tensor:
        return apply_sola(
            infer, self.sola_buffer, self._sola_norm_kernel,
            self._fade_in, self._fade_out,
            self._block_samples, self._sola_buffer_samples, self._sola_search_samples,
        )




class BreathEffect:
    """UV 区域合成气息噪声 — 增加自然气息感，不使用原音频（不漏本音）。

    原理：在 pitchf<=阈值（UV/清音/气息段）叠加带通粉噪声，模拟人声的自然气息。
    噪声是合成的（粉噪声 + 带通滤波），不包含源说话人音色，因此不会"漏本音"。
    """

    def __init__(self):
        self._noise_buffer = None
        self._noise_idx = 0
        self._sr = 0
        self._device = None

    def setup(self, sr: int, device: str, block_samples: int):
        """预生成带通粉噪声 buffer（1 秒，循环使用）。"""
        self._sr = sr
        self._device = device
        noise_len = sr  # 1 秒
        # 粉噪声：白噪声累积和（近似 1/f 频谱）
        white = torch.randn(noise_len, device=device)
        pink = torch.cumsum(white, dim=0)
        pink = pink / (pink.std() + 1e-8)
        # 带通滤波：500Hz ~ 8kHz（模拟口腔气息频谱）
        spec = torch.fft.rfft(pink)
        freqs = torch.fft.rfftfreq(noise_len, 1.0 / sr, device=device)
        # 平滑带通 mask
        low = 500.0
        high = 8000.0
        mask = torch.ones_like(freqs)
        # 低频滚降（500Hz 以下）
        low_mask = torch.sigmoid((freqs - low) / 100.0)
        # 高频滚降（8kHz 以上）
        high_mask = torch.sigmoid((high - freqs) / 500.0)
        mask = low_mask * high_mask
        spec = spec * mask
        pink_filtered = torch.fft.irfft(spec, n=noise_len)
        # 归一化到约 -30dBFS（峰值 0.03）
        peak = pink_filtered.abs().max() + 1e-8
        self._noise_buffer = pink_filtered / peak * 0.03
        self._noise_idx = 0

    def reset(self):
        self._noise_idx = 0

    def process(self, audio: torch.Tensor, pitchf) -> torch.Tensor:
        """在 UV 区域叠加气息噪声。

        Args:
            audio: 输出音频 [samples] GPU tensor
            pitchf: 音高 [1, frames] 或 [frames]（100fps），0=UV
        """
        from rvc.core.experimental import experimental_config
        if (not experimental_config.breath_enabled or experimental_config.breath_strength <= 0
                or self._noise_buffer is None or pitchf is None):
            return audio
        # pitchf 可能是 [1, frames]，压缩到 [frames]
        pf = pitchf.squeeze()
        if pf.dim() == 0:
            return audio
        # UV mask：pitchf <= 阈值
        uv_mask = (pf <= experimental_config.breath_uv_threshold_hz).float()
        # 上采样到音频采样率
        uv_mask_up = torch.nn.functional.interpolate(
            uv_mask[None, None, :], size=audio.shape[0], mode="linear", align_corners=False
        ).squeeze()
        # 取噪声片段（循环使用 buffer）
        n = audio.shape[0]
        if self._noise_idx + n > self._noise_buffer.shape[0]:
            self._noise_idx = 0
        noise = self._noise_buffer[self._noise_idx:self._noise_idx + n]
        self._noise_idx += n
        # 叠加噪声：UV 区域强度高，浊音区域保留低强度底噪（更自然）
        breath = noise * experimental_config.breath_strength * (0.2 + 0.8 * uv_mask_up)
        return audio + breath

class AudioProcessor:
    """音频处理编排器 — 持有降噪 / RMS / SOLA 三个效果器，统一 setup/process/reset。

    输入侧（推理前）：降噪
    输出侧（推理后）：RMS 混合 → SOLA
    """

    def __init__(self):
        self.denoise = DenoiseEffect()
        self.rms_mix = RmsMixEffect()
        self.sola = SolaEffect()
        self.breath = BreathEffect()

    def setup(self, sr: int, block_samples: int, crossfade_samples: int,
              sola_search_samples: int, device: str):
        self.denoise.setup(sr)
        self.rms_mix.setup(sr)
        self.sola.setup(sr, block_samples, crossfade_samples, sola_search_samples, device)
        self.breath.setup(sr, device, block_samples)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用，避免静音数据污染）。"""
        self.denoise.reset()
        self.sola.reset()
        self.breath.reset()

    def process_input(self, mono: torch.Tensor, denoise_enable: bool,
                      denoise_strength: float) -> torch.Tensor:
        """输入侧处理：降噪。"""
        return self.denoise.process(mono, denoise_enable, denoise_strength)

    def process_output(self, infer: torch.Tensor, ref: torch.Tensor,
                       rms_mix: float, is_vc: bool = True) -> torch.Tensor:
        """输出侧处理：RMS 混合 → SOLA。

        is_vc=False（直通模式）时跳过 RMS 混合。
        """
        if is_vc:
            infer = self.rms_mix.process(infer, ref, rms_mix)
        return self.sola.process(infer)
