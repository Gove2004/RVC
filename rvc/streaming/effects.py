"""音频效果器 — RMS 混合 / SOLA / 麦克风气流声，统一 torch.Tensor（GPU）接口。

输出侧处理链：
  RMS 混合 → SOLA 拼接 → 麦克风气流声 → 硬件输出

所有效果器在 setup() 时初始化，热路径纯向量化（零 Python 循环）。
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from rvc.dsp.rms import apply_rms_mix
from rvc.dsp.sola import apply_sola

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
    """SOLA 时间对齐 — 消除块间不连续（相位对齐 + 交叉淡化）。"""

    def __init__(self):
        self.sola_buffer = None
        self._fade_in = None
        self._fade_out = None
        self._sola_norm_kernel = None
        self._block_samples = 0
        self._sola_buffer_samples = 0
        self._sola_search_samples = 0

    def setup(self, sr: int, block_samples: int, sola_buffer_samples: int,
              sola_search_samples: int, device: str):
        self._block_samples = block_samples
        self._sola_buffer_samples = sola_buffer_samples
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


class MicAirflowEffect:
    """麦克风气流声 — 语音振幅调制 + 呼呼声（说话时动态）+ 底噪（静音稳态）。

    呼呼声（纯低频朦胧WongWong，基于wind noise论文）：
    - 仅说话时出现（envelope门控，静音严格=0）
    - 100Hz低频rumble（wind noise能量集中在<100Hz）
    - 75Hz共振层（vortex shedding共振峰，模拟WongWong低沉共鸣）
    - wind缓慢起伏（非稳态，跟随语音节奏）
    底噪：85Hz稳态高斯背景声，无脉冲、无共振、无起伏。

    所有低通均为因果卷积（只左侧pad），避免块末尾输出伸进右pad零区导致能量塌陷。
    块间上下文缓存保证块开头输出有完整历史，无V型凹陷。
    核大小随采样率动态缩放，时间常量/截止频率与sr无关。
    """

    # 增益（不依赖sr）
    BREATH_GAIN = 5.4          # 呼呼声rumble增益（拉满时相对语音≈0dB）
    RESONANCE_GAIN = 0.5       # 共振层增益
    FLOOR_GAIN = 0.6            # 底噪增益（静音可闻）
    MOD_DEPTH = 0.15            # 振幅调制深度（±15%）
    INTENSITY_SCALE = 0.35      # GUI 100% 对应实际 35% 效果
    WIND_MASK_DEPTH = 0.06    # 风掩蔽语音深度

    # @48k 参考核大小（setup中按sr比例缩放）
    _REF_SR = 48000
    _REF_SLOW_K = 2400    # 50ms box低通
    _REF_FLOOR_K = 180    # 85Hz三角核
    _REF_BREATH_K = 150   # 100Hz三角核
    _REF_RESONANCE_K = 200  # 75Hz三角核
    _REF_GATE_LEN = 40    # 10ms EMA平滑

    @staticmethod
    def _triangular_kernel(k: int) -> torch.Tensor:
        """生成三角核（两次移动平均的卷积），频率响应sinc²，-26dB/octave。"""
        size = 2 * k - 1
        idx = torch.arange(size, dtype=torch.float32)
        w = torch.min(idx + 1, size - idx) / (k * k)
        return w.view(1, 1, size)

    def __init__(self):
        self._slow_kernel = None
        self._combo_kernel = None
        self._gate_kernel = None
        self._slow_kernel_gpu = None
        self._combo_kernel_gpu = None
        self._gate_kernel_gpu = None
        self._slow_ctx = 0       # slow核大小-1（上下文缓存长度）
        self._combo_ctx = 0      # combo核大小-1
        self._gate_ctx = 0       # gate核大小-1
        self._mod_norm = 1.0     # box低通后白噪声理论σ=sqrt(1/k)
        self._gate_offset = 0.0  # sigmoid(-2.4)≈0.0832，setup中预计算
        # 块间上下文缓存
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None

    def setup(self, sr: int, device: str = None):
        """根据采样率动态计算所有核大小（时间常量/截止频率与sr无关）。

        Args:
            sr: 采样率
            device: 若提供，将核预迁移到该设备，避免首块热路径H2D同步
        """
        scale = sr / self._REF_SR
        slow_k = max(1, int(self._REF_SLOW_K * scale))
        floor_k = max(1, int(self._REF_FLOOR_K * scale))
        breath_k = max(1, int(self._REF_BREATH_K * scale))
        resonance_k = max(1, int(self._REF_RESONANCE_K * scale))
        gate_len = max(2, int(self._REF_GATE_LEN * scale))
        combo_size = 2 * max(floor_k, breath_k, resonance_k) - 1

        self._slow_kernel = torch.ones(1, 1, slow_k) / slow_k
        # 分组卷积组合核：通道0=breath，通道1=resonance，通道2=floor
        combo = torch.zeros(3, 1, combo_size)
        kb = self._triangular_kernel(breath_k)
        kr = self._triangular_kernel(resonance_k)
        kf = self._triangular_kernel(floor_k)
        combo[0, 0, :kb.shape[2]] = kb[0, 0]
        combo[1, 0, :kr.shape[2]] = kr[0, 0]
        combo[2, 0, :kf.shape[2]] = kf[0, 0]
        self._combo_kernel = combo
        # gate平滑核：指数EMA，反转使最大权重对应最新样本（conv1d是互相关）
        gate_kernel = 0.08 * 0.92 ** torch.arange(gate_len, dtype=torch.float32)
        self._gate_kernel = gate_kernel.flip(0).view(1, 1, -1)

        # 预计算常量（避免热路径GPU同步）
        self._slow_ctx = slow_k - 1
        self._combo_ctx = combo_size - 1
        self._gate_ctx = gate_len - 1
        self._mod_norm = (1.0 / slow_k) ** 0.5  # box低通后白噪声理论σ
        self._gate_offset = float(torch.sigmoid(torch.tensor(-0.08 * 30)).item())

        # 预迁移核到设备（避免首块热路径H2D同步）
        if device is not None:
            self._slow_kernel_gpu = self._slow_kernel.to(device)
            self._combo_kernel_gpu = self._combo_kernel.to(device)
            self._gate_kernel_gpu = self._gate_kernel.to(device)
        else:
            self._slow_kernel_gpu = None
            self._combo_kernel_gpu = None
            self._gate_kernel_gpu = None

    def reset(self):
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None

    @staticmethod
    def _causal_conv1d(x: torch.Tensor, kernel: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """因果1D卷积：只左侧pad(k-1)，输出长度=输入长度。"""
        k = kernel.shape[2]
        x_padded = F.pad(x, (k - 1, 0))
        return F.conv1d(x_padded, kernel, groups=groups)

    def process(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        if intensity <= 0:
            return audio
        intensity = intensity * self.INTENSITY_SCALE

        n = audio.shape[0]
        device = audio.device

        if self._slow_kernel_gpu is None or self._slow_kernel_gpu.device != device:
            self._slow_kernel_gpu = self._slow_kernel.to(device)
            self._combo_kernel_gpu = self._combo_kernel.to(device)
            self._gate_kernel_gpu = self._gate_kernel.to(device)

        # 1. 振幅调制 + 包络：因果卷积 + 跨块累积上下文（自动满长度，不依赖块长≥ctx）
        mod_raw = torch.randn(n, device=device)
        if self._prev_audio is not None:
            audio_ctx = torch.cat([self._prev_audio, audio])
            mod_ctx = torch.cat([self._prev_mod, mod_raw])
        else:
            audio_ctx = audio
            mod_ctx = mod_raw
        # 从拼接后的上下文取最后slow_ctx个，保证累积满长度（首块不足时自动累积）
        self._prev_audio = audio_ctx[-self._slow_ctx:].clone()
        self._prev_mod = mod_ctx[-self._slow_ctx:].clone()

        batch_in = torch.stack([mod_ctx, audio_ctx.abs()], dim=0).unsqueeze(1)
        batch_out = self._causal_conv1d(batch_in, self._slow_kernel_gpu)
        ctx_n = mod_ctx.shape[0]
        mod_low = batch_out[0, 0, ctx_n - n:ctx_n]
        envelope = batch_out[1, 0, ctx_n - n:ctx_n]

        # 振幅调制归一化：用理论σ（box低通后白噪声σ=sqrt(1/k)），避免每块std估计跳变
        mod_low = mod_low / self._mod_norm * self.MOD_DEPTH + 1.0
        mod_low = mod_low.clamp(0.7, 1.3)
        mod_scaled = 1.0 + (mod_low - 1.0) * intensity
        audio_mod = audio * mod_scaled

        # 2. 分组卷积：风噪声（因果卷积 + 跨块累积上下文）
        noise = torch.randn(n, device=device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise])
        else:
            noise_ctx = noise
        self._prev_noise = noise_ctx[-self._combo_ctx:].clone()

        noise_3ch = noise_ctx.view(1, 1, -1).expand(1, 3, -1)
        combo_out = self._causal_conv1d(noise_3ch, self._combo_kernel_gpu, groups=3)
        noise_n = noise_ctx.shape[0]
        breath_low = combo_out[0, 0, noise_n - n:noise_n]
        res_low = combo_out[0, 1, noise_n - n:noise_n]
        floor_low = combo_out[0, 2, noise_n - n:noise_n]

        # 3. 呼呼声门控：sigmoid（减去env=0偏移，静音严格=0）+ 因果EMA平滑
        gate_raw = torch.sigmoid((envelope - 0.08) * 30)
        gate_in = ((gate_raw - self._gate_offset) / (1.0 - self._gate_offset)).clamp(min=0.0)

        # 缓存平滑前的gate（而非平滑后），避免下一块二次滤波
        if self._prev_gate is not None:
            gate_ctx = torch.cat([self._prev_gate, gate_in])
        else:
            gate_ctx = gate_in
        self._prev_gate = gate_ctx[-self._gate_ctx:].clone()

        gate_smooth = self._causal_conv1d(gate_ctx.view(1, 1, -1), self._gate_kernel_gpu)[0, 0]
        envelope_gate = gate_smooth[gate_ctx.shape[0] - n:]

        # wind缓慢起伏（MOD_DEPTH=0时保护，避免0/0）
        if self.MOD_DEPTH > 0:
            wind_mod = 1.0 + 0.4 * (mod_low - 1.0) / self.MOD_DEPTH
        else:
            wind_mod = 1.0

        # 呼呼声 = 100Hz rumble（门控+起伏）+ 75Hz共振（门控）
        rumble = breath_low * envelope_gate * wind_mod * self.BREATH_GAIN
        resonance = res_low * envelope_gate * self.RESONANCE_GAIN
        breath = (rumble + resonance) * intensity

        # 底噪：稳态
        floor = floor_low * intensity * self.FLOOR_GAIN

        # 风掩蔽语音
        wind_mask = 1.0 - self.WIND_MASK_DEPTH * envelope_gate * intensity
        return audio_mod * wind_mask + breath + floor


class AudioProcessor:
    """音频处理编排器 — 持有 RMS / SOLA / 麦克风气流声。

    输出侧处理链：RMS 混合 → SOLA 拼接 → 麦克风气流声
    """

    def __init__(self):
        self.rms_mix = RmsMixEffect()
        self.sola = SolaEffect()
        self.airflow = MicAirflowEffect()

    def setup(self, sr: int, block_samples: int, sola_buffer_samples: int,
              sola_search_samples: int, device: str):
        self.rms_mix.setup(sr)
        self.sola.setup(sr, block_samples, sola_buffer_samples, sola_search_samples, device)
        self.airflow.setup(sr, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用）。"""
        self.sola.reset()
        self.airflow.reset()

    def process_output(self, infer, ref, rms_mix, airflow=0.0):
        """输出侧处理：RMS 混合 → SOLA → 麦克风气流声。

        Args:
            infer: 合成器输出音频
            ref: 原始输入参考（RMS 混合用）
            rms_mix: RMS 混合比例 (0-1)
            airflow: 麦克风气流声强度 (0-1)
        """
        infer = self.rms_mix.process(infer, ref, rms_mix)
        infer = self.sola.process(infer)
        return self.airflow.process(infer, airflow)
