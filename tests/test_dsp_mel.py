"""mel 滤波器与 pad_center 纯函数测试 — S3 新增覆盖（原 train/mel.py 无测试）。

核心约定（librosa 兼容语义，重写必须保持）：
- mel_filter_bank 输出形状 (n_mels, 1 + n_fft//2)，Slaney 归一化下
  每个滤波器积分面积归一（enorm 因子）；
- pad_center 居中填充，奇数差时左侧少补（int 除法截断）。
"""
import unittest

import numpy as np

from rvc.dsp.mel import mel_filter_bank, pad_center


class TestMelFilterBank(unittest.TestCase):
    def test_output_shape(self):
        fb = mel_filter_bank(sr=40000, n_fft=1024, n_mels=128, fmin=0.0, fmax=20000)
        assert fb.shape == (128, 1 + 1024 // 2)

    def test_nonnegative_and_bounded(self):
        fb = mel_filter_bank(sr=48000, n_fft=2048, n_mels=64, fmax=24000)
        assert (fb >= 0).all()
        # 每个滤波器的峰值有限（归一化后不超过 2/带宽量级）
        assert np.isfinite(fb).all()

    def test_slaney_normalization_area(self):
        # Slaney 归一化：相邻 mel 点间距的倒数因子，滤波器积分应接近 1
        sr, n_fft, n_mels = 16000, 400, 20
        fb = mel_filter_bank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=8000)
        # 用 FFT bin 宽度近似积分
        bin_width = (sr / 2) / (1 + n_fft // 2)
        areas = fb.sum(axis=1) * bin_width
        np.testing.assert_allclose(areas, np.ones(n_mels), rtol=0.25)

    def test_deterministic(self):
        kwargs = dict(sr=16000, n_fft=400, n_mels=20, fmax=8000)
        fb1 = mel_filter_bank(**kwargs)
        fb2 = mel_filter_bank(**kwargs)
        np.testing.assert_array_equal(fb1, fb2)
        fb3 = mel_filter_bank(**kwargs, norm=None)
        assert np.isfinite(fb3).all() and (fb3 >= 0).all()


class TestPadCenter(unittest.TestCase):
    def test_centering(self):
        data = np.ones(10)
        out = pad_center(data, size=20)
        assert out.shape == (20,)
        # int((20-10)/2)=5 左补，右侧 5
        assert out[:5].sum() == 0.0
        assert out[5:15].sum() == 10.0
        assert out[15:].sum() == 0.0

    def test_odd_padding_left_short(self):
        data = np.ones(4)
        out = pad_center(data, size=7)
        assert out.shape == (7,)
        # int((7-4)/2)=1 左补，右补 2
        assert out[0] == 0.0 and out[1] == 1.0 and out[-1] == 0.0


if __name__ == "__main__":
    unittest.main()
