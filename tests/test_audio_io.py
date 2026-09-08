"""音频 IO 与工具函数测试 — output_router / wav_io。

覆盖：
- write_main_output: 单声道/立体声写入
- route_secondary_output: 启用/禁用、队列满处理
- write_wav + read_wav_info: FLOAT/PCM_16 往返一致性
- read_wav_info: 非 WAV 文件报错
"""
import os
import queue
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from rvc.audio.output_router import route_secondary_output, write_main_output
from rvc.audio.wav_io import read_audio_info, read_wav_info, write_wav


class TestWriteMainOutput(unittest.TestCase):
    """write_main_output 单声道/立体声写入测试。"""

    def test_mono_output(self):
        """单声道：chunk 写入 outdata[:, 0]。"""
        chunk = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        outdata = np.zeros((3, 1), dtype=np.float32)
        write_main_output(chunk, outdata, channels=1)
        np.testing.assert_allclose(outdata[:, 0], [0.1, 0.2, 0.3], atol=1e-6)

    def test_stereo_output(self):
        """立体声：chunk 扩展为 [:, None] 写入所有通道。"""
        chunk = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        outdata = np.zeros((3, 2), dtype=np.float32)
        write_main_output(chunk, outdata, channels=2)
        np.testing.assert_allclose(outdata[:, 0], [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(outdata[:, 1], [0.1, 0.2, 0.3], atol=1e-6)

    def test_output_shape_preserved(self):
        """写入后 outdata 形状不变。"""
        chunk = torch.randn(100, dtype=torch.float32)
        outdata = np.zeros((100, 2), dtype=np.float32)
        original_shape = outdata.shape
        write_main_output(chunk, outdata, channels=2)
        self.assertEqual(outdata.shape, original_shape)


class TestRouteSecondaryOutput(unittest.TestCase):
    """route_secondary_output 副输出路由测试。"""

    def test_enabled_puts_to_queue(self):
        """启用时：outdata 复制到队列。"""
        outdata = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        out2_q = queue.Queue(maxsize=5)
        route_secondary_output(outdata, stream2=True, out2_q=out2_q, enable_out2=True)
        self.assertFalse(out2_q.empty())
        item = out2_q.get_nowait()
        np.testing.assert_array_equal(item, outdata)

    def test_disabled_does_not_put(self):
        """禁用时：不写入队列。"""
        outdata = np.array([[0.1, 0.2]], dtype=np.float32)
        out2_q = queue.Queue(maxsize=5)
        route_secondary_output(outdata, stream2=True, out2_q=out2_q, enable_out2=False)
        self.assertTrue(out2_q.empty())

    def test_no_stream_does_not_put(self):
        """stream2 为 None 时：不写入队列。"""
        outdata = np.array([[0.1, 0.2]], dtype=np.float32)
        out2_q = queue.Queue(maxsize=5)
        route_secondary_output(outdata, stream2=None, out2_q=out2_q, enable_out2=True)
        self.assertTrue(out2_q.empty())

    def test_full_queue_drops_oldest(self):
        """队列满时：丢弃最旧的一项，再放入新的。"""
        out2_q = queue.Queue(maxsize=2)
        old_data = np.array([[0.0, 0.0]], dtype=np.float32)
        new_data = np.array([[1.0, 1.0]], dtype=np.float32)
        out2_q.put_nowait(old_data)
        out2_q.put_nowait(old_data)  # 队列满
        route_secondary_output(new_data, stream2=True, out2_q=out2_q, enable_out2=True)
        # 队列应该还是满的（2项），最旧的被丢弃
        self.assertEqual(out2_q.qsize(), 2)
        # 最新的应该是 new_data
        item = out2_q.get_nowait()
        np.testing.assert_array_equal(item, old_data)
        item = out2_q.get_nowait()
        np.testing.assert_array_equal(item, new_data)

    def test_copy_not_reference(self):
        """放入队列的是副本，不是引用（修改原数据不影响队列）。"""
        outdata = np.array([[0.1, 0.2]], dtype=np.float32)
        out2_q = queue.Queue(maxsize=5)
        route_secondary_output(outdata, stream2=True, out2_q=out2_q, enable_out2=True)
        outdata[0, 0] = 999.0  # 修改原数据
        item = out2_q.get_nowait()
        self.assertAlmostEqual(item[0, 0], 0.1, places=5)


class TestWavIo(unittest.TestCase):
    """WAV 文件读写测试（临时文件，测试后自动清理）。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav_path(self, name):
        return str(Path(self.tmpdir) / name)

    def test_write_read_float_mono(self):
        """FLOAT32 单声道：写后读元信息一致。"""
        data = np.random.randn(1000).astype(np.float32) * 0.5
        path = self._wav_path("float_mono.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 44100)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["frames"], 1000)
        self.assertEqual(info["bits_per_sample"], 32)
        self.assertAlmostEqual(info["duration"], 1000 / 44100, places=4)

    def test_write_read_float_stereo(self):
        """FLOAT32 立体声：写后读元信息一致。"""
        data = np.random.randn(500, 2).astype(np.float32) * 0.5
        path = self._wav_path("float_stereo.wav")
        write_wav(path, data, samplerate=48000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 48000)
        self.assertEqual(info["channels"], 2)
        self.assertEqual(info["frames"], 500)
        self.assertEqual(info["bits_per_sample"], 32)

    def test_write_read_pcm16_mono(self):
        """PCM16 单声道：写后读元信息一致。"""
        data = np.random.randn(800).astype(np.float32) * 0.3
        path = self._wav_path("pcm16_mono.wav")
        write_wav(path, data, samplerate=22050, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["samplerate"], 22050)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["frames"], 800)
        self.assertEqual(info["bits_per_sample"], 16)

    def test_pcm16_clips_out_of_range(self):
        """PCM16：超出 [-1, 1] 的值被截断。"""
        data = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        path = self._wav_path("pcm16_clip.wav")
        write_wav(path, data, samplerate=16000, subtype="PCM_16")
        info = read_wav_info(path)
        self.assertEqual(info["frames"], 3)  # 不崩溃，帧数正确

    def test_invalid_subtype_raises(self):
        """无效 subtype 抛 ValueError。"""
        data = np.zeros(100, dtype=np.float32)
        path = self._wav_path("invalid.wav")
        with self.assertRaises(ValueError):
            write_wav(path, data, samplerate=16000, subtype="INVALID")

    def test_read_non_wav_raises(self):
        """读取非 WAV 文件抛 ValueError。"""
        path = self._wav_path("not_wav.txt")
        with open(path, "w") as f:
            f.write("not a wav file")
        with self.assertRaises(ValueError):
            read_wav_info(path)

    def test_1d_data_auto_reshape(self):
        """1D 数据自动 reshape 为 (frames, 1)。"""
        data = np.random.randn(200).astype(np.float32)
        path = self._wav_path("1d.wav")
        write_wav(path, data, samplerate=32000, subtype="FLOAT")
        info = read_wav_info(path)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["frames"], 200)

    def test_read_audio_info_wav(self):
        """read_audio_info 对 WAV 文件直接解析（不依赖 ffmpeg）。"""
        data = np.random.randn(300).astype(np.float32) * 0.5
        path = self._wav_path("info.wav")
        write_wav(path, data, samplerate=44100, subtype="FLOAT")
        info = read_audio_info(path)
        self.assertEqual(info["samplerate"], 44100)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["frames"], 300)
        self.assertAlmostEqual(info["duration"], 300 / 44100, places=4)

