"""通用 mel 频谱处理函数（numpy 版）

只依赖 numpy，不依赖 torch。推理侧（RMVPE）与训练侧共用同一实现，
避免两处独立滤波器导致的频谱口径漂移。
"""
from typing import Any

import numpy as np

# HTK mel 刻度（mel = 1127·ln(1 + f/700)）——全仓唯一字面量来源
MEL_HTK_SLOPE = 1127.0
MEL_HTK_F0 = 700.0

# 人声 F0 量程（RMVPE 的输出范围；F0 → mel 归一化边界共用同一公式）
F0_MIN = 50.0   # Hz
F0_MAX = 1100.0  # Hz
F0_MEL_MIN = MEL_HTK_SLOPE * np.log(1.0 + F0_MIN / MEL_HTK_F0)
F0_MEL_MAX = MEL_HTK_SLOPE * np.log(1.0 + F0_MAX / MEL_HTK_F0)


def hz_to_mel_htk(freq, xp=np):
    """Hz → HTK mel。xp 参数接受 numpy 或 torch（两种后端共用同一公式）。"""
    return MEL_HTK_SLOPE * xp.log(1.0 + freq / MEL_HTK_F0)


def _hz_to_mel(freq: np.ndarray | float, htk: bool = False) -> Any:
    """Hz → mel 刻度。

    **两个分支都是 HTK 刻度**，只是写法不同：2595·log10(1+f/700) 与
    1127·ln(1+f/700) 数学等价（2595/ln10 = 1127.0），实测滤波器差异在
    1e-17 量级（65664 个元素里个别点差最后一位，是浮点实现差异而非逻辑差异）。
    保留 htk 参数只为对齐 librosa 的调用签名，本仓两个调用点取值不同
    （RMVPE 传 True、训练侧显式传 True）但产出的滤波器等价。

    注意这与 librosa 的语义**不同**：librosa 的 htk=False 是 Slaney 刻度
    （1kHz 以下线性、以上对数），取点与本函数相差可达数百 Hz。若将来要与
    librosa 默认口径对齐，必须实现 Slaney 分段，不能靠翻转这个布尔值。
    """
    if htk:
        return 2595.0 * np.log10(1.0 + freq / MEL_HTK_F0)
    return MEL_HTK_SLOPE * np.log(1.0 + freq / MEL_HTK_F0)


def _mel_to_hz(mel: np.ndarray | float, htk: bool = False) -> Any:
    """mel 刻度 → Hz（_hz_to_mel 的逆；两个分支都是 HTK，见其说明）。"""
    if htk:
        return MEL_HTK_F0 * (10.0 ** (mel / 2595.0) - 1.0)
    return MEL_HTK_F0 * (np.exp(mel / MEL_HTK_SLOPE) - 1.0)


def _fft_frequencies(sr: float, n_fft: int) -> np.ndarray:
    """FFT 频段中心频率。"""
    return np.linspace(0, sr / 2, 1 + n_fft // 2, dtype=np.float64)


def _mel_frequencies(n_mels: int, fmin: float, fmax: float,
                     htk: bool = False) -> np.ndarray:
    """mel 刻度上均匀分布的频率点。"""
    mel_min = _hz_to_mel(fmin, htk=htk)
    mel_max = _hz_to_mel(fmax, htk=htk)
    mels = np.linspace(mel_min, mel_max, n_mels, dtype=np.float64)
    return _mel_to_hz(mels, htk=htk)


def mel_filter_bank(sr: float, n_fft: int, n_mels: int = 128,
                    fmin: float = 0.0, fmax: float | None = None,
                    htk: bool = False, norm: str | None = "slaney") -> np.ndarray:
    """生成 mel 滤波器矩阵。

    返回 shape: (n_mels, 1 + n_fft//2)

    **兼容性范围（精确表述，别当成「与 librosa 等价」）**：
    - `htk=True` 与 `librosa.filters.mel(..., htk=True)` **逐位一致**（实测
      sr/n_fft/n_mels/fmin/fmax 多组参数下 max|diff| = 0.0）。推理侧 RMVPE
      走的正是这一支，所以官方权重可用。
    - `htk=False` **不与** librosa 默认（Slaney 刻度）一致：librosa 的
      htk=False 是 1kHz 以下线性、以上对数的分段刻度，而本实现的 htk=False
      仍是 HTK 刻度（1127·ln 写法，与 htk=True 数值等价到 1e-17）。实测两者
      取点相差可达 262Hz。
      本仓训练侧（mel_processing.spec_to_mel_torch）已显式传 htk=True，
      与推理侧同支，所以训练/推理口径自洽，只是**与 librosa 默认口径不同**。
      若将来要与 librosa 默认对齐（例如迁移外部用 librosa 预训练的权重），
      必须实现 Slaney 分段，翻转这个布尔值是没用的。
    """
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)

    # FFT 频段中心频率
    fftfreqs = _fft_frequencies(sr, n_fft)

    # mel 频率点（n_mels + 2 个，包含上下边界）
    mel_f = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    # 计算所有 mel 频率点与 FFT 频率点的差（向量化）
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)  # (n_mels+2, n_fft//2+1)

    for i in range(n_mels):
        # 上升沿：从左边界到中心
        lower = -ramps[i] / fdiff[i]
        # 下降沿：从中心到右边界
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney 归一化：让每个滤波器的面积为 1
    if norm == "slaney":
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"mel_filter_bank: 不支持的 norm={norm!r}（仅支持 'slaney' 或 None）")

    return weights

