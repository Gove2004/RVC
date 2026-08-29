"""降噪效果器 — 谱减法（零依赖，GPU 块级处理，零新增延迟）

谱减法：对每帧做 rFFT → 幅谱 → 自适应噪声地板（分频段不对称平滑：
当前幅谱低于地板时地板快速下探跟随噪声，高于地板时极慢上浮，避免锁死在语音上）
→ 软掩码增益 gain = 1 - strength * floor/mag，下限 beta 保护（残余噪声保持
为柔和底噪而非"音乐噪声"）→ irFFT。无需手动"学习噪声"。
"""
import torch

# 谱减参数
_FLOOR_DOWN_COEF = 0.2   # 幅谱低于地板时的下探系数（快）
_FLOOR_UP_FACTOR = 1.0002  # 幅谱高于地板时的上浮系数（慢）
_BETA = 0.1              # 增益下限，防止完全消零造成音乐噪声


class SpectralSubtraction:
    """谱减法降噪 — FFT 频域、自适应噪声地板估计。

    参数:
        strength: 0~1 降噪强度。0 = 直通；越接近 1 压得越狠。
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.strength = 0.0
        self.noise_floor = None  # 每频段的噪声幅谱（首次处理时初始化）

    def set_strength(self, strength: float):
        """设置降噪强度 (0~1)"""
        self.strength = max(0.0, min(1.0, strength))

    def reset(self):
        """清空噪声地板估计（流重启后重新学习当前环境噪声）"""
        self.noise_floor = None

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0:
            return audio

        spec = torch.fft.rfft(audio)
        mag = spec.abs()

        # 自适应噪声地板（首帧直接以当前幅谱初始化）
        floor = self.noise_floor
        if floor is None or floor.shape != mag.shape:
            self.noise_floor = mag.clone()
            floor = self.noise_floor
        below = mag < floor
        if below.any():
            floor[below] += _FLOOR_DOWN_COEF * (mag[below] - floor[below])
        if (~below).any():
            floor[~below] *= _FLOOR_UP_FACTOR

        # 谱减增益：语音段 gain→1，噪声段 gain→1-strength，下限 beta 防音乐噪声
        gain = 1.0 - self.strength * (floor / (mag + 1e-8))
        gain.clamp_(min=_BETA)
        spec *= gain
        return torch.fft.irfft(spec, n=audio.shape[0])

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """调用入口（实时引擎按 nr_ss(mono) 语法调用）。"""
        return self.process(audio)
