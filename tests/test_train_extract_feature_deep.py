"""train/extract_feature 深度测试 — 初始化/停止逻辑/批量处理/特征提取。

覆盖：
- HuBERTExtractor.__init__: 初始化参数存储
- HuBERTExtractor.request_stop: 停止标志设置
- HuBERTExtractor.run: 批量处理（mock 文件系统和模型）
- HuBERTExtractor.extract: 特征提取（mock 模型和音频加载）
"""
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile

import numpy as np
import torch

from rvc.train.extract_feature import HuBERTExtractor


class TestHuBERTExtractorInit(unittest.TestCase):
    """HuBERTExtractor 初始化测试。"""

    @patch("rvc.train.extract_feature.load_hubert")
    def test_creates_extractor(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False)
        self.assertIsNotNone(extractor)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_device_stored(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cuda:0", is_half=False)
        self.assertEqual(extractor.device, "cuda:0")

    @patch("rvc.train.extract_feature.load_hubert")
    def test_is_half_stored(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=True)
        self.assertTrue(extractor.is_half)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_hubert_variant_stored(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False, hubert="chinese")
        self.assertEqual(extractor.hubert, "chinese")

    @patch("rvc.train.extract_feature.load_hubert")
    def test_model_created(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False)
        mock_load.assert_called_once()
        self.assertIsNotNone(extractor.model)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_stop_requested_false_initially(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False)
        self.assertFalse(extractor.stop_requested)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_default_device_cuda(self, mock_load):
        """默认设备是 cuda:0。"""
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor()
        self.assertEqual(extractor.device, "cuda:0")

    @patch("rvc.train.extract_feature.load_hubert")
    def test_default_is_half_true(self, mock_load):
        """默认 is_half 为 True。"""
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor()
        self.assertTrue(extractor.is_half)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_default_hubert_base(self, mock_load):
        """默认 hubert 变体为 base。"""
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor()
        self.assertEqual(extractor.hubert, "base")


class TestHuBERTExtractorStop(unittest.TestCase):
    """HuBERTExtractor 停止逻辑测试。"""

    @patch("rvc.train.extract_feature.load_hubert")
    def test_request_stop_sets_flag(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.request_stop()
        self.assertTrue(extractor.stop_requested)

    @patch("rvc.train.extract_feature.load_hubert")
    def test_request_stop_idempotent(self, mock_load):
        """多次调用 request_stop 不崩溃。"""
        mock_load.return_value = MagicMock()
        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.request_stop()
        extractor.request_stop()
        extractor.request_stop()
        self.assertTrue(extractor.stop_requested)


class TestHuBERTExtractorRun(unittest.TestCase):
    """HuBERTExtractor.run 批量处理测试（mock 文件系统）。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.exp_dir = Path(self.tmpdir) / "exp"
        self.wav_dir = self.exp_dir / "1_16k_wavs"
        self.wav_dir.mkdir(parents=True)

        # 创建几个假 wav 文件
        for i in range(3):
            (self.wav_dir / f"test_{i}.wav").write_bytes(b"fake wav data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_processes_all_files(self, mock_load_audio, mock_load_hubert):
        """run 处理所有 wav 文件。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        count = extractor.run(str(self.exp_dir))
        self.assertEqual(count, 3)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_creates_output_dir(self, mock_load_audio, mock_load_hubert):
        """run 创建输出目录。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir))

        self.assertTrue((self.exp_dir / "3_feature768").exists())

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_progress_callback(self, mock_load_audio, mock_load_hubert):
        """run 调用进度回调。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        progress = []
        def cb(i, total):
            progress.append((i, total))

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir), progress_callback=cb)

        self.assertEqual(len(progress), 3)
        self.assertEqual(progress[0], (1, 3))
        self.assertEqual(progress[2], (3, 3))

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_stop_requested(self, mock_load_audio, mock_load_hubert):
        """stop_requested 为 True 时停止处理。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.request_stop()  # 提前停止
        count = extractor.run(str(self.exp_dir))
        self.assertEqual(count, 3)  # 返回总文件数

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_stop_check_callback(self, mock_load_audio, mock_load_hubert):
        """stop_check 回调返回 True 时停止处理。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        stop_check = MagicMock(return_value=True)
        count = extractor.run(str(self.exp_dir), stop_check=stop_check)
        self.assertEqual(count, 3)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_run_skips_existing_outputs(self, mock_load_audio, mock_load_hubert):
        """已存在输出文件时跳过处理。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        # 预创建第一个文件的输出
        feat_dir = self.exp_dir / "3_feature768"
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.save(feat_dir / "test_0.npy", np.zeros((100, 768), dtype=np.float32))

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir))

        # 第一个文件应该被跳过，只处理后两个
        self.assertEqual(mock_model.call_count, 2)


class TestHuBERTExtractorExtract(unittest.TestCase):
    """HuBERTExtractor.extract 特征提取测试。"""

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_returns_numpy_array(self, mock_load_audio, mock_load_hubert):
        """extract 返回 numpy 数组。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        result = extractor.extract(Path("test.wav"))
        self.assertIsInstance(result, np.ndarray)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_dtype_float32(self, mock_load_audio, mock_load_hubert):
        """extract 返回 float32 数组。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        result = extractor.extract(Path("test.wav"))
        self.assertEqual(result.dtype, np.float32)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_shape_correct(self, mock_load_audio, mock_load_hubert):
        """extract 返回正确形状（frames, 768）。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        result = extractor.extract(Path("test.wav"))
        self.assertEqual(result.shape, (100, 768))

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_calls_model(self, mock_load_audio, mock_load_hubert):
        """extract 调用模型进行推理。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.extract(Path("test.wav"))
        mock_model.assert_called_once()

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_loads_audio_at_16k(self, mock_load_audio, mock_load_hubert):
        """extract 以 16kHz 加载音频。"""
        from rvc.audio.constants import HUBERT_SAMPLE_RATE
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.extract(Path("test.wav"))
        mock_load_audio.assert_called_once_with(Path("test.wav"), HUBERT_SAMPLE_RATE)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_half_precision_when_is_half(self, mock_load_audio, mock_load_hubert):
        """is_half=True 时使用半精度。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=True)
        extractor.extract(Path("test.wav"))
        # 检查输入是否为半精度
        call_args = mock_model.call_args[0][0]
        self.assertEqual(call_args.dtype, torch.float16)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_float_precision_when_not_half(self, mock_load_audio, mock_load_hubert):
        """is_half=False 时使用单精度。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.extract(Path("test.wav"))
        call_args = mock_model.call_args[0][0]
        self.assertEqual(call_args.dtype, torch.float32)

    @patch("rvc.train.extract_feature.load_hubert")
    @patch("rvc.train.extract_feature.load_audio")
    def test_extract_input_shape_1d(self, mock_load_audio, mock_load_hubert):
        """extract 输入形状为 (1, samples)。"""
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output
        mock_load_hubert.return_value = mock_model

        extractor = HuBERTExtractor(device="cpu", is_half=False)
        extractor.extract(Path("test.wav"))
        call_args = mock_model.call_args[0][0]
        self.assertEqual(call_args.shape[0], 1)
        self.assertEqual(call_args.shape[1], 16000)


if __name__ == "__main__":
    unittest.main()
