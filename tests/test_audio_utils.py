"""音频工具函数测试 — RMS、Slicer、normalize_audio。"""
import unittest

import numpy as np

from rvc.train.preprocess import _rms, Slicer, normalize_audio


class TestRMS(unittest.TestCase):
    """_rms 均方根计算测试。"""

    def test_output_shape(self):
        """输出帧数应正确。"""
        y = np.zeros(1000, dtype=np.float32)
        rms = _rms(y, frame_length=256, hop_length=128)
        # center=True + reflect padding，帧数约为 len(y)/hop_length + 1
        self.assertGreater(len(rms), 0)
        self.assertLessEqual(len(rms), len(y) // 128 + 2)

    def test_silence_returns_zero(self):
        """全零输入应返回全零 RMS。"""
        y = np.zeros(1000, dtype=np.float32)
        rms = _rms(y, frame_length=256, hop_length=128)
        self.assertTrue(np.all(rms == 0))

    def test_sine_wave_rms(self):
        """正弦波的 RMS 应接近 1/sqrt(2) ≈ 0.707。"""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        y = np.sin(2 * np.pi * 440 * t)
        rms = _rms(y, frame_length=1024, hop_length=512)
        # 中间帧（不受边界影响）应接近 0.707
        mid = rms[len(rms) // 2]
        self.assertAlmostEqual(mid, 1 / np.sqrt(2), places=2)

    def test_non_negative(self):
        """RMS 应非负。"""
        y = np.random.randn(1000).astype(np.float32)
        rms = _rms(y, frame_length=256, hop_length=128)
        self.assertTrue(np.all(rms >= 0))

    def test_dtype(self):
        """输出应为 float32。"""
        y = np.zeros(1000, dtype=np.float32)
        rms = _rms(y, frame_length=256, hop_length=128)
        self.assertEqual(rms.dtype, np.float32)

    def test_constant_signal(self):
        """常数信号的 RMS 应等于该常数的绝对值。"""
        y = np.full(1000, 0.5, dtype=np.float32)
        rms = _rms(y, frame_length=256, hop_length=128)
        mid = rms[len(rms) // 2]
        self.assertAlmostEqual(mid, 0.5, places=3)


class TestSlicer(unittest.TestCase):
    """Slicer 音频切片测试。"""

    def test_short_audio_returns_original(self):
        """短于 min_length 的音频应返回原音频。"""
        slicer = Slicer(sr=16000, min_length=1500)  # 1.5s
        wav = np.zeros(1000, dtype=np.float32)  # 0.0625s
        result = slicer.slice(wav)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.array_equal(result[0], wav))

    def test_silence_returns_empty_or_single(self):
        """全静音音频应返回空列表或单个片段。"""
        slicer = Slicer(sr=16000, threshold=-42.0, min_length=1500)
        wav = np.zeros(16000 * 3, dtype=np.float32)  # 3s 静音
        result = slicer.slice(wav)
        # 全静音可能返回空列表或一个全静音片段
        self.assertIsInstance(result, list)

    def test_speech_with_silence_gap(self):
        """有静音间隔的语音应被切成多段。"""
        sr = 16000
        slicer = Slicer(sr=sr, threshold=-42.0, min_length=500, min_interval=300, max_sil_kept=200)
        # 构造：0.5s 语音 + 1s 静音 + 0.5s 语音
        speech1 = np.random.randn(sr // 2).astype(np.float32) * 0.5
        silence = np.zeros(sr, dtype=np.float32)
        speech2 = np.random.randn(sr // 2).astype(np.float32) * 0.5
        wav = np.concatenate([speech1, silence, speech2])
        result = slicer.slice(wav)
        # 应该至少切成 2 段
        self.assertGreaterEqual(len(result), 2)

    def test_output_are_arrays(self):
        """输出应为 numpy 数组列表。"""
        slicer = Slicer(sr=16000)
        wav = np.random.randn(16000).astype(np.float32) * 0.1
        result = slicer.slice(wav)
        for seg in result:
            self.assertIsInstance(seg, np.ndarray)

    def test_no_empty_segments(self):
        """不应返回过短的片段（小于 hop_size）。"""
        slicer = Slicer(sr=16000, hop_size=15)
        wav = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        result = slicer.slice(wav)
        min_hop = int(16000 * 15 / 1000)
        for seg in result:
            self.assertGreater(len(seg), min_hop)


class TestNormalizeAudio(unittest.TestCase):
    """normalize_audio 音频归一化测试。"""

    def test_silence_returns_silence(self):
        """全零输入应返回全零。"""
        wav = np.zeros(1000, dtype=np.float32)
        result = normalize_audio(wav)
        self.assertTrue(np.all(result == 0))
        self.assertEqual(result.dtype, np.float32)

    def test_peak_too_high_returns_empty(self):
        """峰值 > 2.5 应返回空数组。"""
        wav = np.full(1000, 3.0, dtype=np.float32)
        result = normalize_audio(wav)
        self.assertEqual(len(result), 0)

    def test_normal_audio_normalized(self):
        """正常音频应被归一化（混合归一化：75%峰值归一化 + 25%原信号）。"""
        wav = np.full(1000, 2.0, dtype=np.float32)  # 峰值 2.0 < 2.5
        result = normalize_audio(wav)
        # 公式：wav/peak*0.9*0.75 + wav*0.25
        # 对于常数 2.0：2.0/2.0*0.675 + 2.0*0.25 = 0.675 + 0.5 = 1.175
        self.assertAlmostEqual(np.abs(result).max(), 1.175, places=2)

    def test_output_dtype(self):
        """输出应为 float32。"""
        wav = np.random.randn(1000).astype(np.float64)
        result = normalize_audio(wav)
        self.assertEqual(result.dtype, np.float32)

    def test_peak_at_boundary(self):
        """峰值刚好 2.5 应正常处理（不返回空）。"""
        wav = np.full(1000, 2.5, dtype=np.float32)
        result = normalize_audio(wav)
        self.assertGreater(len(result), 0)

    def test_small_signal_normalized_to_07(self):
        """小信号也会被归一化到约 0.7 峰值。"""
        wav = np.full(1000, 0.1, dtype=np.float32)
        result = normalize_audio(wav)
        # normalize_audio 公式：wav/peak*0.9*0.75 + wav*0.25
        # 对于常数信号 0.1：0.1/0.1*0.9*0.75 + 0.1*0.25 = 0.675 + 0.025 = 0.7
        self.assertAlmostEqual(np.abs(result).max(), 0.7, places=1)


if __name__ == "__main__":
    unittest.main()
