"""mel 滤波器和 pad_center 测试 — 验证从 librosa 迁移后的实现正确性。"""
import unittest

import numpy as np

from rvc.audio.mel import mel_filter_bank, pad_center, _hz_to_mel, _mel_to_hz, _fft_frequencies, _mel_frequencies


class TestMelFilterBank(unittest.TestCase):
    """mel_filter_bank 基本性质测试。"""

    def test_output_shape(self):
        """输出形状应为 (n_mels, 1 + n_fft//2)。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=80)
        self.assertEqual(weights.shape, (80, 513))

    def test_non_negative(self):
        """所有滤波器权重应非负。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=80)
        self.assertTrue(np.all(weights >= 0))

    def test_triangular_shape(self):
        """每个滤波器应为三角形（单峰，两侧递减）。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=40, fmin=0, fmax=8000)
        for i in range(weights.shape[0]):
            row = weights[i]
            if row.sum() == 0:
                continue
            peak = np.argmax(row)
            # 峰值左侧应非递减（允许 0）
            left = row[:peak + 1]
            self.assertTrue(np.all(np.diff(left) >= -1e-6), f"filter {i} left side not monotonic")
            # 峰值右侧应非递增
            right = row[peak:]
            self.assertTrue(np.all(np.diff(right) <= 1e-6), f"filter {i} right side not monotonic")

    def test_slaney_normalization(self):
        """Slaney 归一化后，高频滤波器权重应更小（带宽更宽）。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=80, norm="slaney")
        # 低频滤波器的峰值应大于高频滤波器（因为低频带宽更窄，归一化后权重更大）
        low_peak = weights[10].max()
        high_peak = weights[70].max()
        self.assertGreater(low_peak, high_peak)

    def test_slaney_normalization(self):
        """Slaney 归一化后，高频滤波器权重应更小（带宽更宽）。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=80, norm="slaney")
        # 低频滤波器的峰值应大于高频滤波器（因为低频带宽更窄，归一化后权重更大）
        low_peak = weights[10].max()
        high_peak = weights[70].max()
        self.assertGreater(low_peak, high_peak)

    def test_fmin_fmax(self):
        """fmin/fmax 应限制滤波器的频率范围。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=40, fmin=100, fmax=4000)
        # 低于 fmin 的频段应几乎没有权重
        fftfreqs = _fft_frequencies(16000, 1024)
        low_band = fftfreqs < 100
        self.assertTrue(weights[:, low_band].sum() < 1e-6)

    def test_dtype(self):
        """输出应为 float32。"""
        weights = mel_filter_bank(sr=16000, n_fft=1024, n_mels=80)
        self.assertEqual(weights.dtype, np.float32)


class TestPadCenter(unittest.TestCase):
    """pad_center 测试。"""

    def test_output_length(self):
        """输出长度应为 size。"""
        data = np.array([1, 2, 3])
        padded = pad_center(data, 7)
        self.assertEqual(len(padded), 7)

    def test_centered(self):
        """数据应居中。"""
        data = np.array([1, 2, 3])
        padded = pad_center(data, 7)
        # [0, 0, 1, 2, 3, 0, 0]
        self.assertTrue(np.array_equal(padded, [0, 0, 1, 2, 3, 0, 0]))

    def test_odd_size_difference(self):
        """奇数长度差时，左侧少补一个。"""
        data = np.array([1, 2, 3])
        padded = pad_center(data, 6)
        # [0, 1, 2, 3, 0, 0] - 左侧 1 个，右侧 2 个
        self.assertEqual(padded[0], 0)
        self.assertEqual(padded[1], 1)
        self.assertEqual(padded[4], 0)

    def test_same_size(self):
        """size 等于数据长度时应返回原数据。"""
        data = np.array([1, 2, 3])
        padded = pad_center(data, 3)
        self.assertTrue(np.array_equal(padded, data))


class TestMelConversions(unittest.TestCase):
    """Hz ↔ mel 转换测试。"""

    def test_hz_to_mel_to_hz_roundtrip(self):
        """Hz → mel → Hz 应往返一致。"""
        for hz in [0, 100, 1000, 4000, 8000]:
            mel = _hz_to_mel(hz)
            hz_back = _mel_to_hz(mel)
            self.assertAlmostEqual(hz, hz_back, places=3)

    def test_htk_roundtrip(self):
        """HTK 公式也应往返一致。"""
        for hz in [0, 100, 1000, 4000, 8000]:
            mel = _hz_to_mel(hz, htk=True)
            hz_back = _mel_to_hz(mel, htk=True)
            self.assertAlmostEqual(hz, hz_back, places=3)

    def test_mel_monotonic(self):
        """mel 刻度应随频率单调递增。"""
        freqs = [100, 200, 500, 1000, 2000, 4000, 8000]
        mels = [_hz_to_mel(f) for f in freqs]
        for i in range(len(mels) - 1):
            self.assertLess(mels[i], mels[i + 1])

    def test_fft_frequencies(self):
        """FFT 频率应从 0 到 sr/2。"""
        freqs = _fft_frequencies(16000, 1024)
        self.assertEqual(freqs[0], 0)
        self.assertAlmostEqual(freqs[-1], 8000)
        self.assertEqual(len(freqs), 513)

    def test_mel_frequencies_count(self):
        """mel 频率点数量应正确。"""
        freqs = _mel_frequencies(10, fmin=0, fmax=8000)
        self.assertEqual(len(freqs), 10)
        self.assertAlmostEqual(freqs[0], 0)
        self.assertAlmostEqual(freqs[-1], 8000)


if __name__ == "__main__":
    unittest.main()
