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


class AudioProcessor:
    """音频处理编排器 — 持有降噪 / RMS / SOLA 三个效果器，统一 setup/process/reset。

    输入侧（推理前）：降噪
    输出侧（推理后）：RMS 混合 → SOLA
    """

    def __init__(self):
        self.denoise = DenoiseEffect()
        self.rms_mix = RmsMixEffect()
        self.sola = SolaEffect()

    def setup(self, sr: int, block_samples: int, crossfade_samples: int,
              sola_search_samples: int, device: str):
        self.denoise.setup(sr)
        self.rms_mix.setup(sr)
        self.sola.setup(sr, block_samples, crossfade_samples, sola_search_samples, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用，避免静音数据污染）。"""
        self.denoise.reset()
        self.sola.reset()

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
