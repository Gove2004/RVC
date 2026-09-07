"""通用 mel 频谱处理函数（numpy 版）

从 rvc.train.mel_processing 拆出，供推理侧（RMVPE）和训练侧共同使用。
只依赖 numpy，不依赖 torch。
"""
import numpy as np


def _hz_to_mel(freq, htk=False):
    """Hz → mel 刻度。"""
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _mel_to_hz(mel, htk=False):
    """mel 刻度 → Hz。"""
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _fft_frequencies(sr, n_fft):
    """FFT 频段中心频率。"""
    return np.linspace(0, sr / 2, 1 + n_fft // 2, dtype=np.float64)


def _mel_frequencies(n_mels, fmin=0.0, fmax=None, htk=False):
    """mel 刻度上均匀分布的频率点。"""
    if fmax is None:
        fmax = 11025.0  # librosa 默认 fmax=sr/2，但调用方会传入
    mel_min = _hz_to_mel(fmin, htk=htk)
    mel_max = _hz_to_mel(fmax, htk=htk)
    mels = np.linspace(mel_min, mel_max, n_mels, dtype=np.float64)
    return _mel_to_hz(mels, htk=htk)


def mel_filter_bank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney"):
    """生成 mel 滤波器矩阵（兼容 librosa.filters.mel）。

    返回 shape: (n_mels, 1 + n_fft//2)
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

    return weights


def pad_center(data, size):
    """居中填充数组到指定长度（兼容 librosa.util.pad_center）。"""
    n = data.shape[-1]
    lpad = int((size - n) / 2)
    rpad = size - n - lpad
    return np.pad(data, (lpad, rpad), mode="constant")
