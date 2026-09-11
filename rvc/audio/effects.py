"""音频效果器 — RMS 混合 / SOLA 时间对齐 / 空气感高频提升，统一 torch.Tensor（GPU）接口。

RealtimeEngine._cb_impl 不再直接调用各底层函数，而是通过 AudioProcessor 编排：
  输出侧：process_output(infer, ref, config) → RMS 混合 → 空气感 → SOLA

所有效果器在 setup() 时初始化（与 _init_processing 同时），process() 时零分配。
"""
import logging
import math

import numpy as np
import torch

from rvc.audio.realtime_mix import apply_rms_mix
from rvc.audio.sola import apply_sola

logger = logging.getLogger(__name__)


@torch.jit.script
def _biquad_filter_block(audio: torch.Tensor, b0: float, b1: float, b2: float,
                          a1: float, a2: float, x1: float, x2: float,
                          y1: float, y2: float):
    """BiQuad Direct Form I 滤波（JIT 编译，GPU 上单 kernel 执行）。

    Returns:
        output: 滤波后的音频
        x1, x2, y1, y2: 更新后的延迟线状态
    """
    n = audio.shape[0]
    output = torch.empty_like(audio)
    for i in range(n):
        x0 = audio[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[i] = y0
        x2 = x1
        x1 = x0.item()
        y2 = y1
        y1 = y0.item()
    return output, x1, x2, y1, y2




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


class AirPresenceEffect:
    """空气感效果器 — High Shelf 高频搁架提升，增加通透感和空间感。

    提升 8kHz 以上的高频，让声音更亮、更有"空气感"。
    采用 BiQuad 二阶搁架滤波器，参数平滑插值避免爆音。
    强度 0~100 对应 0~+12dB 增益。

    滤波核心用 torch.jit.script 编译，GPU 上单 kernel 执行，
    避免 Python 循环逐元素索引的开销。
    """

    CUTOFF_HZ = 8000.0
    Q = 0.707  # 巴特沃斯 Q，平滑过渡
    MAX_GAIN_DB = 12.0  # 强度 100 对应 +12dB

    def __init__(self):
        # 滤波器系数（标量，a0 归一化为 1）
        self._b0 = 1.0
        self._b1 = 0.0
        self._b2 = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        # 滤波器延迟线（Direct Form I，Python float，避免 GPU 标量同步）
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        # 当前增益（用于平滑）
        self._current_gain_db = 0.0
        self._target_gain_db = 0.0
        self._sr = 48000

    def setup(self, sr: int, block_samples: int, device: str):
        """初始化滤波器系数和状态。"""
        self._sr = sr
        self._current_gain_db = 0.0
        self._target_gain_db = 0.0
        self._compute_coeffs(0.0)
        self.reset()

    def _compute_coeffs(self, gain_db: float):
        """计算 BiQuad High Shelf 滤波器系数（参考 W3C Audio EQ Cookbook）。"""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * self.CUTOFF_HZ / self._sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * self.Q)
        sqrt_A = math.sqrt(A)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def reset(self):
        """重置滤波器延迟线和增益状态（切换模型/停止时调用）。"""
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._current_gain_db = 0.0
        self._target_gain_db = 0.0
        self._compute_coeffs(0.0)

    def process(self, audio: torch.Tensor, strength: float) -> torch.Tensor:
        """处理音频块。

        Args:
            audio: 输入音频 (N,) GPU tensor
            strength: 强度 0.0~100.0（0=关闭，100=最大 +12dB）

        Returns:
            处理后的音频（新张量）
        """
        if strength <= 0.0:
            return audio

        # 目标增益：0~100 映射到 0~+12dB
        self._target_gain_db = strength * (self.MAX_GAIN_DB / 100.0)

        # 参数平滑（每块更新 5%，避免爆音）
        gain_diff = self._target_gain_db - self._current_gain_db
        if abs(gain_diff) > 0.01:
            self._current_gain_db += gain_diff * 0.05
            self._compute_coeffs(self._current_gain_db)

        # JIT 编译的 BiQuad 滤波（GPU 上单 kernel）
        output, x1, x2, y1, y2 = _biquad_filter_block(
            audio, self._b0, self._b1, self._b2,
            self._a1, self._a2,
            self._x1, self._x2, self._y1, self._y2,
        )
        self._x1, self._x2 = x1, x2
        self._y1, self._y2 = y1, y2

        return output


class AudioProcessor:
    """音频处理编排器 — 持有 RMS / 空气感 / SOLA 三个效果器，统一 setup/process/reset。

    输出侧（推理后）：RMS 混合 → 空气感 → SOLA
    """

    def __init__(self):
        self.rms_mix = RmsMixEffect()
        self.air_presence = AirPresenceEffect()
        self.sola = SolaEffect()

    def setup(self, sr: int, block_samples: int, crossfade_samples: int,
              sola_search_samples: int, device: str):
        self.rms_mix.setup(sr)
        self.air_presence.setup(sr, block_samples, device)
        self.sola.setup(sr, block_samples, crossfade_samples, sola_search_samples, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用，避免静音数据污染）。"""
        self.air_presence.reset()
        self.sola.reset()

    def process_output(self, infer: torch.Tensor, ref: torch.Tensor,
                       rms_mix: float, is_vc: bool = True,
                       air_presence: float = 0.0) -> torch.Tensor:
        """输出侧处理：RMS 混合 → 空气感 → SOLA。

        is_vc=False（直通模式）时跳过 RMS 混合。
        """
        if is_vc:
            infer = self.rms_mix.process(infer, ref, rms_mix)
        infer = self.air_presence.process(infer, air_presence)
        return self.sola.process(infer)
