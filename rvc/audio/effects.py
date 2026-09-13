"""音频效果器 — RMS 混合 / SOLA 时间对齐，统一 torch.Tensor（GPU）接口。

RealtimeEngine._cb_impl 不再直接调用各底层函数，而是通过 AudioProcessor 编排：
  输出侧：process_output(infer, ref, config) → RMS 混合 → 清辅音保护 → SOLA

所有效果器在 setup() 时初始化（与 _init_processing 同时），process() 时零分配。
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from rvc.audio.realtime_mix import apply_rms_mix
from rvc.audio.sola import apply_sola

logger = logging.getLogger(__name__)


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
    """音频处理编排器 — 持有 RMS / SOLA 两个效果器，统一 setup/process/reset。

    输出侧（推理后）：RMS 混合 → 清辅音保护 → SOLA
    """

    def __init__(self):
        self.rms_mix = RmsMixEffect()
        self.sola = SolaEffect()
        self.sr = 48000

    def setup(self, sr: int, block_samples: int, crossfade_samples: int,
              sola_search_samples: int, device: str):
        self.sr = sr
        self.rms_mix.setup(sr)
        self.sola.setup(sr, block_samples, crossfade_samples, sola_search_samples, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用，避免静音数据污染）。"""
        self.sola.reset()

    def _apply_consonant_protection(self, infer, ref, pitchf, protect, protect_threshold_hz):
        """清辅音保护：输出侧混合原始输入音频。

        对 F0 低于阈值的清音帧，用原始输入音频替换合成输出，避免清辅音
        被过度转换产生电音/撕裂。

        Args:
            infer: 合成输出音频 [N]
            ref: 原始输入音频 [M]（M >= N）
            pitchf: 连续 F0 [T]，100fps
            protect: 保护强度（0-1，0=不保护，1=完全用原始输入）
            protect_threshold_hz: 清浊判定阈值（Hz）

        Returns:
            保护后的音频 [N]
        """
        if pitchf is None or pitchf.shape[0] == 0 or protect <= 0:
            return infer

        n = infer.shape[0]
        samples_per_frame = self.sr // 100

        # F0 上采样到音频采样率
        pitchf_up = pitchf.repeat_interleave(samples_per_frame)
        if pitchf_up.shape[0] >= n:
            pitchf_up = pitchf_up[:n]
        else:
            pitchf_up = F.pad(pitchf_up, (0, n - pitchf_up.shape[0]))

        # 生成清音 mask
        uv_mask = (pitchf_up < protect_threshold_hz).float()

        # 平滑 mask（5ms 移动平均，避免边界突变）
        smooth_win = max(1, self.sr // 200)
        uv_mask = F.avg_pool1d(
            uv_mask[None, None, :],
            kernel_size=smooth_win, stride=1, padding=smooth_win // 2,
        )[0, 0, :n]

        # 混合：清音帧偏向原始输入
        ref_aligned = ref[:n] if ref.shape[0] >= n else F.pad(ref, (0, n - ref.shape[0]))
        strength = protect * uv_mask
        infer = infer * (1 - strength) + ref_aligned * strength
        return infer

    def process_output(self, infer, ref, rms_mix, is_vc=True,
                       pitchf=None, protect=0.0, protect_threshold_hz=25.0):
        """输出侧处理：RMS 混合 → 清辅音保护 → SOLA。

        is_vc=False（直通模式）时跳过 RMS 混合和清辅音保护。
        """
        if is_vc:
            infer = self.rms_mix.process(infer, ref, rms_mix)
            infer = self._apply_consonant_protection(
                infer, ref, pitchf, protect, protect_threshold_hz,
            )
        return self.sola.process(infer)
