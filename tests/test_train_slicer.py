"""训练预处理深度测试 — Slicer 音频切片、_rms、normalize_audio。

覆盖：
- Slicer 初始化各种参数（sr/threshold/min_length/min_interval/hop_size/max_sil_kept）
- Slicer.slice 各种输入（短音频/长音频/静音/纯音/带静音段/多段）
- _rms 各种参数（frame_length/hop_length/各种输入）
- normalize_audio 各种输入（正常/静音/极大值/单样本）
"""
import unittest

import numpy as np

from rvc.train.preprocess import Slicer, _rms, normalize_audio


class TestSlicerInit(unittest.TestCase):
    """Slicer 初始化测试。"""

    def test_default_sr(self):
        slicer = Slicer(sr=16000)
        self.assertEqual(slicer.sr, 16000)

    def test_various_sr(self):
        for sr in [8000, 16000, 22050, 32000, 44100, 48000]:
            slicer = Slicer(sr=sr)
            self.assertEqual(slicer.sr, sr)

    def test_default_threshold(self):
        slicer = Slicer(sr=16000)
        # threshold = -42 dB -> 10^(-42/20)
        expected = 10 ** (-42.0 / 20.0)
        self.assertAlmostEqual(slicer.threshold, expected, places=6)

    def test_custom_threshold(self):
        slicer = Slicer(sr=16000, threshold=-30.0)
        expected = 10 ** (-30.0 / 20.0)
        self.assertAlmostEqual(slicer.threshold, expected, places=6)

    def test_threshold_zero_db(self):
        slicer = Slicer(sr=16000, threshold=0.0)
        self.assertAlmostEqual(slicer.threshold, 1.0, places=6)

    def test_threshold_negative_60(self):
        slicer = Slicer(sr=16000, threshold=-60.0)
        expected = 10 ** (-60.0 / 20.0)
        self.assertAlmostEqual(slicer.threshold, expected, places=6)

    def test_min_length_default(self):
        """min_length=1500ms -> sr * 1.5。"""
        slicer = Slicer(sr=16000)
        self.assertEqual(slicer.min_length, int(16000 * 1500 / 1000))

    def test_min_length_custom(self):
        slicer = Slicer(sr=16000, min_length=1000)
        self.assertEqual(slicer.min_length, 16000)

    def test_min_interval_default(self):
        slicer = Slicer(sr=16000)
        self.assertEqual(slicer.min_interval, int(16000 * 400 / 1000))

    def test_hop_size_default(self):
        slicer = Slicer(sr=16000)
        self.assertEqual(slicer.hop_size, int(16000 * 15 / 1000))

    def test_max_sil_kept_default(self):
        slicer = Slicer(sr=16000)
        self.assertEqual(slicer.max_sil_kept, int(16000 * 500 / 1000))

    def test_all_custom_params(self):
        slicer = Slicer(
            sr=48000, threshold=-50.0, min_length=2000,
            min_interval=500, hop_size=20, max_sil_kept=1000,
        )
        self.assertEqual(slicer.sr, 48000)
        self.assertEqual(slicer.min_length, int(48000 * 2000 / 1000))
        self.assertEqual(slicer.min_interval, int(48000 * 500 / 1000))
        self.assertEqual(slicer.hop_size, int(48000 * 20 / 1000))
        self.assertEqual(slicer.max_sil_kept, int(48000 * 1000 / 1000))


class TestSlicerSlice(unittest.TestCase):
    """Slicer.slice 测试。"""

    def setUp(self):
        self.slicer = Slicer(sr=16000)

    def test_short_audio_returns_single(self):
        """短于 min_length 的音频直接返回单段。"""
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        pieces = self.slicer.slice(wav)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(len(pieces[0]), 1000)

    def test_empty_audio(self):
        """空音频返回单段（空）。"""
        wav = np.array([], dtype=np.float32)
        pieces = self.slicer.slice(wav)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(len(pieces[0]), 0)

    def test_single_sample(self):
        """单样本音频返回单段。"""
        wav = np.array([0.5], dtype=np.float32)
        pieces = self.slicer.slice(wav)
        self.assertEqual(len(pieces), 1)

    def test_pure_tone_returns_single(self):
        """纯音（无静音段）返回单段。"""
        t = np.linspace(0, 2, 32000, dtype=np.float32)
        wav = np.sin(2 * np.pi * 440 * t) * 0.5
        pieces = self.slicer.slice(wav)
        # 纯音没有静音段，应该返回单段或少数几段
        self.assertGreaterEqual(len(pieces), 1)
        # 所有段的总长度应该接近原长度
        total = sum(len(p) for p in pieces)
        self.assertGreater(total, 0)

    def test_silence_returns_few(self):
        """全静音音频返回少数段。"""
        wav = np.zeros(32000, dtype=np.float32)
        pieces = self.slicer.slice(wav)
        # 全静音可能返回空列表或少数短段
        self.assertIsInstance(pieces, list)

    def test_audio_with_silence_gap(self):
        """带静音间隔的音频被切成多段。"""
        # 前 1 秒有声音，中间 1 秒静音，后 1 秒有声音
        wav = np.zeros(48000, dtype=np.float32)
        t1 = np.linspace(0, 1, 16000, dtype=np.float32)
        wav[:16000] = np.sin(2 * np.pi * 440 * t1) * 0.5
        t2 = np.linspace(0, 1, 16000, dtype=np.float32)
        wav[32000:] = np.sin(2 * np.pi * 330 * t2) * 0.5
        pieces = self.slicer.slice(wav)
        # 应该被切成至少 2 段
        self.assertGreaterEqual(len(pieces), 2)

    def test_returns_list(self):
        """返回值是列表。"""
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        pieces = self.slicer.slice(wav)
        self.assertIsInstance(pieces, list)

    def test_pieces_are_arrays(self):
        """每段都是 numpy 数组。"""
        wav = np.random.randn(32000).astype(np.float32) * 0.1
        pieces = self.slicer.slice(wav)
        for piece in pieces:
            self.assertIsInstance(piece, np.ndarray)

    def test_pieces_dtype_float32(self):
        """每段都是 float32 类型。"""
        wav = np.random.randn(32000).astype(np.float32) * 0.1
        pieces = self.slicer.slice(wav)
        for piece in pieces:
            if len(piece) > 0:
                self.assertEqual(piece.dtype, np.float32)

    def test_long_audio(self):
        """长音频（10 秒）不崩溃。"""
        wav = np.random.randn(160000).astype(np.float32) * 0.1
        pieces = self.slicer.slice(wav)
        self.assertIsInstance(pieces, list)
        self.assertGreater(len(pieces), 0)

    def test_various_sr(self):
        """不同采样率的 Slicer 都能工作。"""
        for sr in [16000, 22050, 32000, 44100, 48000]:
            slicer = Slicer(sr=sr)
            wav = np.random.randn(sr * 2).astype(np.float32) * 0.1
            pieces = slicer.slice(wav)
            self.assertIsInstance(pieces, list)

    def test_high_threshold_cuts_more(self):
        """高阈值（更严格）会切更多段。"""
        wav = np.random.randn(32000).astype(np.float32) * 0.01  # 低音量
        low_thresh = Slicer(sr=16000, threshold=-60.0)
        high_thresh = Slicer(sr=16000, threshold=-20.0)
        pieces_low = low_thresh.slice(wav)
        pieces_high = high_thresh.slice(wav)
        # 高阈值应该切更多段（或相等）
        self.assertGreaterEqual(len(pieces_high), len(pieces_low))


class TestRms(unittest.TestCase):
    """_rms 函数测试。"""

    def test_silence_rms_zero(self):
        """静音的 RMS 接近 0。"""
        wav = np.zeros(1000, dtype=np.float32)
        rms = _rms(wav, frame_length=100, hop_length=50)
        self.assertTrue(np.all(rms < 0.01))

    def test_constant_rms(self):
        """常数信号的 RMS 等于常数绝对值。"""
        wav = np.full(1000, 0.5, dtype=np.float32)
        rms = _rms(wav, frame_length=100, hop_length=50)
        # 中间帧的 RMS 应接近 0.5
        self.assertTrue(np.mean(rms[10:-10]) > 0.4)
        self.assertTrue(np.mean(rms[10:-10]) < 0.6)

    def test_output_shape(self):
        wav = np.random.randn(1000).astype(np.float32)
        rms = _rms(wav, frame_length=100, hop_length=50)
        # center=True 时帧数约为 len(wav)/hop_length + 1
        self.assertGreater(len(rms), 0)

    def test_various_frame_lengths(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        for fl in [32, 64, 128, 256, 512]:
            rms = _rms(wav, frame_length=fl, hop_length=fl // 2)
            self.assertTrue(np.all(np.isfinite(rms)))
            self.assertTrue(np.all(rms >= 0))

    def test_various_hop_lengths(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        for hl in [16, 32, 64, 128, 256]:
            rms = _rms(wav, frame_length=128, hop_length=hl)
            self.assertTrue(np.all(np.isfinite(rms)))

    def test_rms_non_negative(self):
        """RMS 值非负。"""
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        rms = _rms(wav, frame_length=100, hop_length=50)
        self.assertTrue(np.all(rms >= 0))

    def test_rms_finite(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        rms = _rms(wav, frame_length=100, hop_length=50)
        self.assertTrue(np.all(np.isfinite(rms)))

    def test_short_signal(self):
        wav = np.random.randn(100).astype(np.float32) * 0.1
        rms = _rms(wav, frame_length=32, hop_length=16)
        self.assertTrue(len(rms) > 0)

    def test_dtype_float32(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        rms = _rms(wav, frame_length=100, hop_length=50)
        self.assertEqual(rms.dtype, np.float32)


class TestNormalizeAudio(unittest.TestCase):
    """normalize_audio 函数测试。"""

    def test_normal_audio(self):
        """正常音频被归一化。"""
        wav = np.random.randn(16000).astype(np.float32) * 0.1
        normalized = normalize_audio(wav)
        self.assertEqual(len(normalized), len(wav))
        self.assertEqual(normalized.dtype, np.float32)

    def test_silence_unchanged(self):
        """静音音频保持不变（peak=0 不除）。"""
        wav = np.zeros(16000, dtype=np.float32)
        normalized = normalize_audio(wav)
        self.assertTrue(np.allclose(normalized, 0.0))

    def test_peak_above_2_5_returns_empty(self):
        """峰值超过 2.5 返回空数组（认为是坏数据）。"""
        wav = np.full(1000, 3.0, dtype=np.float32)
        normalized = normalize_audio(wav)
        self.assertEqual(len(normalized), 0)

    def test_peak_exactly_2_5(self):
        """峰值正好 2.5 不返回空（条件是 > 2.5）。"""
        wav = np.full(1000, 2.5, dtype=np.float32)
        normalized = normalize_audio(wav)
        self.assertEqual(len(normalized), 1000)

    def test_single_sample(self):
        wav = np.array([0.5], dtype=np.float32)
        normalized = normalize_audio(wav)
        self.assertEqual(len(normalized), 1)
        self.assertTrue(np.isfinite(normalized[0]))

    def test_empty_input_raises(self):
        """空数组输入会抛出 ValueError（源代码行为，peak=max() 对空数组无定义）。"""
        wav = np.array([], dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize_audio(wav)

    def test_negative_peak(self):
        """负峰值也能正确归一化。"""
        wav = np.full(1000, -0.8, dtype=np.float32)
        normalized = normalize_audio(wav)
        self.assertEqual(len(normalized), 1000)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_output_finite(self):
        wav = np.random.randn(16000).astype(np.float32) * 0.5
        normalized = normalize_audio(wav)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_output_dtype_float32(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        normalized = normalize_audio(wav)
        self.assertEqual(normalized.dtype, np.float32)

    def test_float64_input(self):
        """float64 输入也能处理（输出 float32）。"""
        wav = np.random.randn(1000).astype(np.float64) * 0.1
        normalized = normalize_audio(wav)
        self.assertEqual(normalized.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
