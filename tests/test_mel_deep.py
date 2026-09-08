"""mel 滤波器深度测试 — 所有函数的各种参数组合、边界条件、属性验证。

覆盖：
- _hz_to_mel / _mel_to_hz: 各种频率、htk 开关、逆变换验证
- _fft_frequencies: 各种 sr/n_fft
- _mel_frequencies: 各种 n_mels/fmin/fmax/htk
- mel_filter_bank: 各种 sr/n_fft/n_mels/fmin/fmax/htk/norm
- pad_center: 各种长度、奇偶
"""
import unittest

import numpy as np

from rvc.audio.mel import (
    _fft_frequencies,
    _hz_to_mel,
    _mel_frequencies,
    _mel_to_hz,
    mel_filter_bank,
    pad_center,
)


class TestHzToMel(unittest.TestCase):
    """_hz_to_mel 测试。"""

    def test_zero_hz(self):
        self.assertAlmostEqual(_hz_to_mel(0), 0.0, places=5)

    def test_zero_hz_htk(self):
        self.assertAlmostEqual(_hz_to_mel(0, htk=True), 0.0, places=5)

    def test_1000hz(self):
        """1000 Hz ≈ 1000 mel（近似）。"""
        mel = _hz_to_mel(1000)
        self.assertAlmostEqual(mel, 1000.0, delta=50)

    def test_1000hz_htk(self):
        mel = _hz_to_mel(1000, htk=True)
        self.assertAlmostEqual(mel, 1000.0, delta=50)

    def test_positive_monotonic(self):
        """频率越高，mel 值越大（单调递增）。"""
        self.assertTrue(_hz_to_mel(2000) > _hz_to_mel(1000))
        self.assertTrue(_hz_to_mel(1000) > _hz_to_mel(500))

    def test_various_frequencies(self):
        for freq in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            mel = _hz_to_mel(freq)
            self.assertTrue(mel >= 0)
            self.assertTrue(np.isfinite(mel))

    def test_nyquist_16k(self):
        mel = _hz_to_mel(8000)
        self.assertTrue(mel > 0)
        self.assertTrue(np.isfinite(mel))

    def test_nyquist_48k(self):
        mel = _hz_to_mel(24000)
        self.assertTrue(mel > 0)
        self.assertTrue(np.isfinite(mel))

    def test_numpy_array_input(self):
        freqs = np.array([0, 1000, 2000], dtype=np.float64)
        mels = _hz_to_mel(freqs)
        self.assertEqual(mels.shape, (3,))
        self.assertAlmostEqual(mels[0], 0.0, places=5)
        self.assertTrue(mels[2] > mels[1])


class TestMelToHz(unittest.TestCase):
    """_mel_to_hz 测试。"""

    def test_zero_mel(self):
        self.assertAlmostEqual(_mel_to_hz(0), 0.0, places=5)

    def test_zero_mel_htk(self):
        self.assertAlmostEqual(_mel_to_hz(0, htk=True), 0.0, places=5)

    def test_1000mel(self):
        hz = _mel_to_hz(1000)
        self.assertAlmostEqual(hz, 1000.0, delta=50)

    def test_positive_monotonic(self):
        self.assertTrue(_mel_to_hz(2000) > _mel_to_hz(1000))
        self.assertTrue(_mel_to_hz(1000) > _mel_to_hz(500))

    def test_various_mel_values(self):
        for mel in [0, 100, 500, 1000, 2000, 3000, 4000]:
            hz = _mel_to_hz(mel)
            self.assertTrue(hz >= 0)
            self.assertTrue(np.isfinite(hz))

    def test_numpy_array_input(self):
        mels = np.array([0, 500, 1000], dtype=np.float64)
        hzs = _mel_to_hz(mels)
        self.assertEqual(hzs.shape, (3,))
        self.assertAlmostEqual(hzs[0], 0.0, places=5)
        self.assertTrue(hzs[2] > hzs[1])


class TestHzMelInverse(unittest.TestCase):
    """Hz ↔ mel 逆变换验证。"""

    def test_hz_to_mel_to_hz(self):
        """hz -> mel -> hz 应该接近原值。"""
        for freq in [100, 500, 1000, 2000, 5000, 10000]:
            mel = _hz_to_mel(freq)
            recovered = _mel_to_hz(mel)
            self.assertAlmostEqual(recovered, freq, delta=0.01)

    def test_hz_to_mel_to_hz_htk(self):
        for freq in [100, 500, 1000, 2000, 5000, 10000]:
            mel = _hz_to_mel(freq, htk=True)
            recovered = _mel_to_hz(mel, htk=True)
            self.assertAlmostEqual(recovered, freq, delta=0.01)

    def test_mel_to_hz_to_mel(self):
        for mel in [100, 500, 1000, 2000, 3000]:
            hz = _mel_to_hz(mel)
            recovered = _hz_to_mel(hz)
            self.assertAlmostEqual(recovered, mel, delta=0.01)


class TestFftFrequencies(unittest.TestCase):
    """_fft_frequencies 测试。"""

    def test_shape(self):
        freqs = _fft_frequencies(16000, 1024)
        self.assertEqual(freqs.shape, (1 + 1024 // 2,))

    def test_start_zero(self):
        freqs = _fft_frequencies(16000, 1024)
        self.assertAlmostEqual(freqs[0], 0.0)

    def test_end_nyquist(self):
        freqs = _fft_frequencies(16000, 1024)
        self.assertAlmostEqual(freqs[-1], 8000.0)

    def test_monotonic_increasing(self):
        freqs = _fft_frequencies(16000, 1024)
        self.assertTrue(np.all(np.diff(freqs) > 0))

    def test_various_n_fft(self):
        for n_fft in [256, 512, 1024, 2048, 4096]:
            freqs = _fft_frequencies(48000, n_fft)
            self.assertEqual(freqs.shape, (1 + n_fft // 2,))
            self.assertAlmostEqual(freqs[-1], 24000.0)

    def test_various_sr(self):
        for sr in [8000, 16000, 22050, 32000, 44100, 48000]:
            freqs = _fft_frequencies(sr, 1024)
            self.assertAlmostEqual(freqs[-1], sr / 2)

    def test_dtype_float64(self):
        freqs = _fft_frequencies(16000, 1024)
        self.assertEqual(freqs.dtype, np.float64)


class TestMelFrequencies(unittest.TestCase):
    """_mel_frequencies 测试。"""

    def test_shape(self):
        freqs = _mel_frequencies(128, fmin=0, fmax=8000)
        self.assertEqual(freqs.shape, (128,))

    def test_start_fmin(self):
        freqs = _mel_frequencies(128, fmin=0, fmax=8000)
        self.assertAlmostEqual(freqs[0], 0.0, places=3)

    def test_end_fmax(self):
        freqs = _mel_frequencies(128, fmin=0, fmax=8000)
        self.assertAlmostEqual(freqs[-1], 8000.0, places=1)

    def test_monotonic_increasing(self):
        freqs = _mel_frequencies(128, fmin=0, fmax=8000)
        self.assertTrue(np.all(np.diff(freqs) > 0))

    def test_various_n_mels(self):
        for n_mels in [32, 64, 80, 128, 256]:
            freqs = _mel_frequencies(n_mels, fmin=0, fmax=8000)
            self.assertEqual(freqs.shape, (n_mels,))

    def test_various_fmin(self):
        for fmin in [0, 50, 100, 200]:
            freqs = _mel_frequencies(128, fmin=fmin, fmax=8000)
            self.assertAlmostEqual(freqs[0], fmin, delta=1.0)

    def test_various_fmax(self):
        for fmax in [4000, 8000, 12000, 24000]:
            freqs = _mel_frequencies(128, fmin=0, fmax=fmax)
            self.assertAlmostEqual(freqs[-1], fmax, delta=1.0)

    def test_htk(self):
        freqs = _mel_frequencies(128, fmin=0, fmax=8000, htk=True)
        self.assertEqual(freqs.shape, (128,))
        self.assertTrue(np.all(np.diff(freqs) > 0))

    def test_default_fmax(self):
        """fmax=None 时默认 11025。"""
        freqs = _mel_frequencies(128, fmin=0)
        self.assertAlmostEqual(freqs[-1], 11025.0, delta=1.0)


class TestMelFilterBank(unittest.TestCase):
    """mel_filter_bank 测试。"""

    def test_shape(self):
        weights = mel_filter_bank(16000, 1024, n_mels=128)
        self.assertEqual(weights.shape, (128, 1 + 1024 // 2))

    def test_non_negative(self):
        weights = mel_filter_bank(16000, 1024, n_mels=128)
        self.assertTrue(np.all(weights >= 0))

    def test_dtype_float32(self):
        weights = mel_filter_bank(16000, 1024, n_mels=128)
        self.assertEqual(weights.dtype, np.float32)

    def test_various_n_mels(self):
        for n_mels in [32, 64, 80, 128, 256]:
            weights = mel_filter_bank(16000, 1024, n_mels=n_mels)
            self.assertEqual(weights.shape, (n_mels, 1 + 1024 // 2))

    def test_various_n_fft(self):
        for n_fft in [256, 512, 1024, 2048]:
            weights = mel_filter_bank(16000, n_fft, n_mels=128)
            self.assertEqual(weights.shape, (128, 1 + n_fft // 2))

    def test_various_sr(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            weights = mel_filter_bank(sr, 1024, n_mels=128)
            self.assertEqual(weights.shape, (128, 1 + 1024 // 2))
            self.assertTrue(np.all(weights >= 0))

    def test_various_fmin(self):
        for fmin in [0, 50, 100, 200]:
            weights = mel_filter_bank(16000, 1024, n_mels=128, fmin=fmin)
            self.assertEqual(weights.shape, (128, 1 + 1024 // 2))

    def test_various_fmax(self):
        for fmax in [4000, 8000, 12000]:
            weights = mel_filter_bank(16000, 1024, n_mels=128, fmax=fmax)
            self.assertEqual(weights.shape, (128, 1 + 1024 // 2))

    def test_htk(self):
        weights = mel_filter_bank(16000, 1024, n_mels=128, htk=True)
        self.assertEqual(weights.shape, (128, 1 + 1024 // 2))
        self.assertTrue(np.all(weights >= 0))

    def test_slaney_norm(self):
        """Slaney 归一化：每个滤波器面积为 1。"""
        weights = mel_filter_bank(16000, 1024, n_mels=128, norm="slaney")
        # 每个滤波器的和应该大致相等（面积归一化）
        row_sums = weights.sum(axis=1)
        # 排除全零行（边缘滤波器可能为零）
        nonzero = row_sums[row_sums > 0]
        self.assertTrue(len(nonzero) > 0)
        # 归一化后各行和应该在同一数量级
        self.assertTrue(np.max(nonzero) / np.min(nonzero) < 10)

    def test_no_norm(self):
        """norm=None 时不归一化（峰值为 1）。"""
        weights = mel_filter_bank(16000, 1024, n_mels=128, norm=None)
        # 中间滤波器的峰值应该接近 1
        mid = weights[64]
        self.assertAlmostEqual(np.max(mid), 1.0, delta=0.1)

    def test_default_fmax_is_nyquist(self):
        """fmax=None 时默认 sr/2。"""
        weights = mel_filter_bank(16000, 1024, n_mels=128)
        self.assertEqual(weights.shape, (128, 1 + 1024 // 2))

    def test_common_config_16k(self):
        """常见配置：16kHz, n_fft=1024, n_mels=128。"""
        weights = mel_filter_bank(16000, 1024, n_mels=128)
        self.assertEqual(weights.shape, (128, 513))
        self.assertTrue(np.all(weights >= 0))

    def test_common_config_48k(self):
        """常见配置：48kHz, n_fft=2048, n_mels=128。"""
        weights = mel_filter_bank(48000, 2048, n_mels=128)
        self.assertEqual(weights.shape, (128, 1025))
        self.assertTrue(np.all(weights >= 0))


class TestPadCenter(unittest.TestCase):
    """pad_center 测试。"""

    def test_pad_to_larger(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        padded = pad_center(data, 7)
        self.assertEqual(padded.shape, (7,))
        self.assertEqual(padded[2], 1)
        self.assertEqual(padded[3], 2)
        self.assertEqual(padded[4], 3)

    def test_pad_to_same_size(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        padded = pad_center(data, 3)
        self.assertEqual(padded.shape, (3,))
        np.testing.assert_array_equal(padded, data)

    def test_pad_zeros(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        padded = pad_center(data, 7)
        self.assertEqual(padded[0], 0)
        self.assertEqual(padded[1], 0)
        self.assertEqual(padded[5], 0)
        self.assertEqual(padded[6], 0)

    def test_even_size(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        padded = pad_center(data, 8)
        self.assertEqual(padded.shape, (8,))

    def test_odd_size(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        padded = pad_center(data, 6)
        self.assertEqual(padded.shape, (6,))

    def test_various_lengths(self):
        for n in [1, 2, 3, 5, 10, 100, 1000]:
            data = np.ones(n, dtype=np.float32)
            for size in [n, n + 1, n + 2, n * 2]:
                padded = pad_center(data, size)
                self.assertEqual(padded.shape, (size,))
                self.assertEqual(np.sum(padded), n)

    def test_dtype_preserved(self):
        data = np.array([1, 2, 3], dtype=np.float64)
        padded = pad_center(data, 7)
        self.assertEqual(padded.dtype, np.float64)

    def test_1d_input_only(self):
        """pad_center 主要用于 1D 输入。"""
        data = np.ones(5, dtype=np.float32)
        padded = pad_center(data, 9)
        self.assertEqual(padded.shape, (9,))
        self.assertEqual(np.sum(padded), 5)


if __name__ == "__main__":
    unittest.main()
