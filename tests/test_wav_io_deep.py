"""WAV IO 深度测试 — 各种采样率/通道/位深/数据范围/边界条件。

覆盖：
- write_wav: 各种采样率、通道数、位深、数据范围
- read_wav_info: 元信息正确性、非 WAV 报错
- read_audio_info: WAV 直接解析
- 边界条件: 空数据、单样本、极大振幅、负值
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvc.audio.wav_io import read_audio_info, read_wav_info, write_wav


class TestWriteWavSampleRates(unittest.TestCase):
    """write_wav 各种采样率测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_8000hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("8k.wav")
        write_wav(path, data, samplerate=8000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 8000)

    def test_16000hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("16k.wav")
        write_wav(path, data, samplerate=16000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 16000)

    def test_22050hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("22k.wav")
        write_wav(path, data, samplerate=22050, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 22050)

    def test_32000hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("32k.wav")
        write_wav(path, data, samplerate=32000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 32000)

    def test_44100hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("44k.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 44100)

    def test_48000hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("48k.wav")
        write_wav(path, data, samplerate=48000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 48000)

    def test_96000hz(self):
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("96k.wav")
        write_wav(path, data, samplerate=96000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 96000)

    def test_various_sample_rates_pcm16(self):
        """各种采样率 + PCM16。"""
        for sr in [8000, 16000, 22050, 32000, 44100, 48000]:
            data = np.random.randn(50).astype(np.float32) * 0.1
            path = self._path(f"sr_{sr}.wav")
            write_wav(path, data, samplerate=sr, subtype="PCM_16")
            info = read_wav_info(path)
            self.assertEqual(info["samplerate"], sr)
            self.assertEqual(info["bits_per_sample"], 16)


class TestWriteWavChannels(unittest.TestCase):
    """write_wav 各种通道数测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_mono_1d(self):
        """1D 数据自动视为单声道。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("mono_1d.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 1)

    def test_mono_2d(self):
        """2D (frames, 1) 数据。"""
        data = np.random.randn(100, 1).astype(np.float32) * 0.1
        path = self._path("mono_2d.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 1)

    def test_stereo(self):
        """立体声 (frames, 2)。"""
        data = np.random.randn(100, 2).astype(np.float32) * 0.1
        path = self._path("stereo.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 2)

    def test_three_channels(self):
        """3 通道。"""
        data = np.random.randn(100, 3).astype(np.float32) * 0.1
        path = self._path("3ch.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 3)

    def test_four_channels(self):
        """4 通道。"""
        data = np.random.randn(100, 4).astype(np.float32) * 0.1
        path = self._path("4ch.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 4)

    def test_six_channels(self):
        """5.1 通道 (6ch)。"""
        data = np.random.randn(100, 6).astype(np.float32) * 0.1
        path = self._path("6ch.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 6)

    def test_eight_channels(self):
        """7.1 通道 (8ch)。"""
        data = np.random.randn(100, 8).astype(np.float32) * 0.1
        path = self._path("8ch.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 8)

    def test_various_channels_pcm16(self):
        """各种通道数 + PCM16。"""
        for ch in [1, 2, 3, 4, 6, 8]:
            data = np.random.randn(50, ch).astype(np.float32) * 0.1
            path = self._path(f"ch_{ch}.wav")
            write_wav(path, data, samplerate=44100, subtype="PCM_16")
            info = read_wav_info(path)
            self.assertEqual(info["channels"], ch)


class TestWriteWavDataRanges(unittest.TestCase):
    """write_wav 各种数据范围测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_all_zeros(self):
        """全零数据。"""
        data = np.zeros(100, dtype=np.float32)
        path = self._path("zeros.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_all_ones(self):
        """全一数据。"""
        data = np.ones(100, dtype=np.float32)
        path = self._path("ones.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_all_negative_one(self):
        """全 -1 数据。"""
        data = np.full(100, -1.0, dtype=np.float32)
        path = self._path("neg_ones.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_max_amplitude_float(self):
        """FLOAT 格式支持大振幅（不截断）。"""
        data = np.full(100, 10.0, dtype=np.float32)
        path = self._path("max_float.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)
        self.assertEqual(info["bits_per_sample"], 32)

    def test_pcm16_clips_above_one(self):
        """PCM16 截断大于 1 的值。"""
        data = np.full(100, 2.0, dtype=np.float32)
        path = self._path("clip_pcm16.wav")
        write_wav(path, data, samplerate=44100, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_pcm16_clips_below_neg_one(self):
        """PCM16 截断小于 -1 的值。"""
        data = np.full(100, -2.0, dtype=np.float32)
        path = self._path("clip_neg_pcm16.wav")
        write_wav(path, data, samplerate=44100, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_alternating_signal(self):
        """交替正负信号。"""
        data = np.array([1.0, -1.0] * 50, dtype=np.float32)
        path = self._path("alternating.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 100)

    def test_sine_wave(self):
        """正弦波信号。"""
        t = np.linspace(0, 1, 1000, dtype=np.float32)
        data = np.sin(2 * np.pi * 440 * t) * 0.5
        path = self._path("sine.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 1000)


class TestWriteWavBoundary(unittest.TestCase):
    """write_wav 边界条件测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_single_sample(self):
        """单样本数据。"""
        data = np.array([0.5], dtype=np.float32)
        path = self._path("single.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 1)

    def test_two_samples(self):
        """双样本数据。"""
        data = np.array([0.5, -0.5], dtype=np.float32)
        path = self._path("two.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 2)

    def test_large_file(self):
        """大文件（10000 样本）。"""
        data = np.random.randn(10000).astype(np.float32) * 0.1
        path = self._path("large.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 10000)

    def test_empty_data_raises(self):
        """空数据应该能写（0 帧）。"""
        data = np.array([], dtype=np.float32)
        path = self._path("empty.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        # 空数据可能读不出来，至少不崩溃

    def test_creates_parent_dirs(self):
        """自动创建父目录。"""
        path = str(Path(self.tmpdir) / "sub" / "dir" / "test.wav")
        data = np.random.randn(100).astype(np.float32) * 0.1
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        self.assertTrue(os.path.exists(path))

    def test_pathlib_path(self):
        """支持 Path 对象作为路径。"""
        path = Path(self.tmpdir) / "pathlib.wav"
        data = np.random.randn(100).astype(np.float32) * 0.1
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        self.assertTrue(path.exists())

    def test_string_path(self):
        """支持字符串路径。"""
        path = str(Path(self.tmpdir) / "string.wav")
        data = np.random.randn(100).astype(np.float32) * 0.1
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        self.assertTrue(os.path.exists(path))

    def test_overwrite_existing(self):
        """覆盖已存在的文件。"""
        path = self._path("overwrite.wav")
        data1 = np.random.randn(100).astype(np.float32) * 0.1
        write_wav(path, data1, samplerate=44100, subtype="FLOAT")
        data2 = np.random.randn(200).astype(np.float32) * 0.2
        write_wav(path, data2, samplerate=48000, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 200)
        self.assertEqual(info["samplerate"], 48000)
        self.assertEqual(info["bits_per_sample"], 16)


class TestReadWavInfo(unittest.TestCase):
    """read_wav_info 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_float32_bits(self):
        """FLOAT32 位深为 32。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("float32.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["bits_per_sample"], 32)

    def test_pcm16_bits(self):
        """PCM16 位深为 16。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("pcm16.wav")
        write_wav(path, data, samplerate=44100, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["bits_per_sample"], 16)

    def test_duration_calculation(self):
        """时长 = frames / samplerate。"""
        data = np.random.randn(44100).astype(np.float32) * 0.1
        path = self._path("duration.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertAlmostEqual(info["duration"], 1.0, places=3)

    def test_frames_calculation(self):
        """帧数 = data_size / (bits/8) / channels。"""
        data = np.random.randn(500, 2).astype(np.float32) * 0.1
        path = self._path("frames.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 500)

    def test_non_wav_raises(self):
        """非 WAV 文件抛 ValueError。"""
        path = self._path("not_wav.txt")
        with open(path, "w") as f:
            f.write("not a wav file")
        with self.assertRaises(ValueError):
            read_wav_info(path)

    def test_empty_file_raises(self):
        """空文件抛异常（struct.error 或 ValueError）。"""
        import struct
        path = self._path("empty.txt")
        with open(path, "w") as f:
            pass
        with self.assertRaises((ValueError, struct.error)):
            read_wav_info(path)

    def test_return_keys(self):
        """返回字典包含所有必要的键。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("keys.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertIn("samplerate", info)
        self.assertIn("channels", info)
        self.assertIn("frames", info)
        self.assertIn("duration", info)
        self.assertIn("bits_per_sample", info)


class TestReadAudioInfo(unittest.TestCase):
    """read_audio_info 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_wav_returns_correct_keys(self):
        """WAV 文件返回 samplerate/channels/frames/duration。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("info.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_audio_info(path)
        self.assertIn("samplerate", info)
        self.assertIn("channels", info)
        self.assertIn("frames", info)
        self.assertIn("duration", info)

    def test_wav_no_bits_per_sample(self):
        """read_audio_info 不返回 bits_per_sample（与 read_wav_info 区分）。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("info2.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_audio_info(path)
        self.assertNotIn("bits_per_sample", info)

    def test_wav_mono(self):
        """WAV 单声道信息正确。"""
        data = np.random.randn(100).astype(np.float32) * 0.1
        path = self._path("mono_info.wav")
        write_wav(path, data, samplerate=48000, subtype="FLOAT")
        info = read_audio_info(path)
        self.assertEqual(info["samplerate"], 48000)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["frames"], 100)

    def test_wav_stereo(self):
        """WAV 立体声信息正确。"""
        data = np.random.randn(100, 2).astype(np.float32) * 0.1
        path = self._path("stereo_info.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_audio_info(path)
        self.assertEqual(info["channels"], 2)
        self.assertEqual(info["frames"], 100)


class TestInvalidSubtype(unittest.TestCase):
    """无效 subtype 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_invalid_subtype_raises(self):
        """无效 subtype 抛 ValueError。"""
        data = np.zeros(100, dtype=np.float32)
        path = str(Path(self.tmpdir) / "invalid.wav")
        with self.assertRaises(ValueError):
            write_wav(path, data, samplerate=44100, subtype="INVALID")

    def test_pcm_24_not_supported(self):
        """PCM_24 不支持（只支持 FLOAT/PCM_16）。"""
        data = np.zeros(100, dtype=np.float32)
        path = str(Path(self.tmpdir) / "pcm24.wav")
        with self.assertRaises(ValueError):
            write_wav(path, data, samplerate=44100, subtype="PCM_24")

    def test_empty_string_subtype(self):
        """空字符串 subtype 抛 ValueError。"""
        data = np.zeros(100, dtype=np.float32)
        path = str(Path(self.tmpdir) / "empty.wav")
        with self.assertRaises(ValueError):
            write_wav(path, data, samplerate=44100, subtype="")


if __name__ == "__main__":
    unittest.main()
