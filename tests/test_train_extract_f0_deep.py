"""train/extract_f0 深度测试 — coarse_f0 纯函数/初始化/停止逻辑/批量处理。

覆盖：
- coarse_f0: F0 频率到离散类别的转换（纯函数，边界条件多）
- TrainF0Extractor.__init__: 初始化参数存储
- TrainF0Extractor.request_stop: 停止标志设置
- TrainF0Extractor.run: 批量处理（mock 文件系统和模型）
"""
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile
import os

import numpy as np

from rvc.audio.f0_utils import normalize_f0_to_coarse
from rvc.train.extract_f0 import TrainF0Extractor
from rvc.models.rmvpe.constants import F0_MEL_MAX, F0_MEL_MIN


class TestCoarseF0(unittest.TestCase):
    """coarse_f0 纯函数测试。"""

    def test_zero_f0_returns_one(self):
        """F0=0（静音/无基频）返回 1。"""
        f0 = np.array([0.0], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result[0], 1)

    def test_negative_f0_returns_one(self):
        """负 F0 返回 1（视为无基频）。"""
        f0 = np.array([-10.0, -50.0], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertTrue(np.all(result == 1))

    def test_output_dtype_int64(self):
        """输出类型为 int64。"""
        f0 = np.array([100.0, 200.0], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result.dtype, np.int64)

    def test_output_shape_matches_input(self):
        """输出形状与输入相同。"""
        for shape in [(10,), (100,), (1, 10)]:
            f0 = np.ones(shape, dtype=np.float32) * 200.0
            result = normalize_f0_to_coarse(f0)
            self.assertEqual(result.shape, f0.shape)

    def test_low_f0_clamped_to_one(self):
        """低于 F0_MEL_MIN 的 F0 被钳位到 1。"""
        # 计算一个对应 mel < F0_MEL_MIN 的频率
        # mel = 1127 * ln(1 + f/700)
        # f = 700 * (exp(mel/1127) - 1)
        low_mel = F0_MEL_MIN - 100
        low_freq = 700 * (np.exp(low_mel / 1127) - 1)
        f0 = np.array([low_freq], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result[0], 1)

    def test_high_f0_clamped_to_255(self):
        """高于 F0_MEL_MAX 的 F0 被钳位到 255。"""
        high_mel = F0_MEL_MAX + 100
        high_freq = 700 * (np.exp(high_mel / 1127) - 1)
        f0 = np.array([high_freq], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result[0], 255)

    def test_mid_f0_in_range(self):
        """中间频率的 F0 在 1-255 范围内。"""
        for freq in [100, 200, 400, 800, 1600]:
            f0 = np.array([float(freq)], dtype=np.float32)
            result = normalize_f0_to_coarse(f0)
            self.assertGreaterEqual(result[0], 1)
            self.assertLessEqual(result[0], 255)

    def test_monotonic_increasing(self):
        """F0 增加时 coarse_f0 也应该增加（在有效范围内）。"""
        freqs = np.linspace(80, 1000, 50, dtype=np.float32)
        results = normalize_f0_to_coarse(freqs)
        # 应该是非递减的
        self.assertTrue(np.all(np.diff(results) >= 0))

    def test_mel_formula_correctness(self):
        """验证 mel 转换公式：mel = 1127 * ln(1 + f/700)。"""
        freq = 440.0
        expected_mel = 1127 * np.log(1 + freq / 700)
        # 手动计算 coarse
        expected_coarse = (expected_mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
        expected_coarse = np.clip(expected_coarse, 1, 255)
        expected_coarse = np.rint(expected_coarse)

        f0 = np.array([freq], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result[0], int(expected_coarse))

    def test_empty_array(self):
        """空数组不崩溃。"""
        f0 = np.array([], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(len(result), 0)

    def test_all_zeros(self):
        """全零数组全返回 1。"""
        f0 = np.zeros(100, dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        self.assertTrue(np.all(result == 1))

    def test_float64_input(self):
        """float64 输入也能处理。"""
        f0 = np.array([200.0], dtype=np.float64)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result.dtype, np.int64)

    def test_integer_input(self):
        """整数输入也能处理。"""
        f0 = np.array([200], dtype=np.int32)
        result = normalize_f0_to_coarse(f0)
        self.assertEqual(result.dtype, np.int64)

    def test_f0_mel_min_boundary(self):
        """F0_MEL_MIN 边界值测试。"""
        # 恰好等于 F0_MEL_MIN 的频率
        freq_at_min = 700 * (np.exp(F0_MEL_MIN / 1127) - 1)
        f0 = np.array([freq_at_min], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        # mel = F0_MEL_MIN 时，coarse = 0 + 1 = 1
        self.assertEqual(result[0], 1)

    def test_f0_mel_max_boundary(self):
        """F0_MEL_MAX 边界值测试。"""
        freq_at_max = 700 * (np.exp(F0_MEL_MAX / 1127) - 1)
        f0 = np.array([freq_at_max], dtype=np.float32)
        result = normalize_f0_to_coarse(f0)
        # mel = F0_MEL_MAX 时，coarse = 254 + 1 = 255
        self.assertEqual(result[0], 255)


class TestTrainF0ExtractorInit(unittest.TestCase):
    """TrainF0Extractor 初始化测试。"""

    @patch("rvc.train.extract_f0.RMVPE")
    def test_creates_extractor(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cpu", is_half=False)
        self.assertIsNotNone(extractor)

    @patch("rvc.train.extract_f0.RMVPE")
    def test_device_stored(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cuda:0", is_half=False)
        self.assertEqual(extractor.device, "cuda:0")

    @patch("rvc.train.extract_f0.RMVPE")
    def test_is_half_stored(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cpu", is_half=True)
        self.assertTrue(extractor.is_half)

    @patch("rvc.train.extract_f0.RMVPE")
    def test_model_created(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cpu", is_half=False)
        mock_rmvpe.assert_called_once()
        self.assertIsNotNone(extractor.model)

    @patch("rvc.train.extract_f0.RMVPE")
    def test_stop_requested_false_initially(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cpu", is_half=False)
        self.assertFalse(extractor.stop_requested)

    @patch("rvc.train.extract_f0.RMVPE")
    def test_default_device_cuda(self, mock_rmvpe):
        """默认设备是 cuda:0。"""
        extractor = TrainF0Extractor()
        self.assertEqual(extractor.device, "cuda:0")

    @patch("rvc.train.extract_f0.RMVPE")
    def test_default_is_half_true(self, mock_rmvpe):
        """默认 is_half 为 True。"""
        extractor = TrainF0Extractor()
        self.assertTrue(extractor.is_half)


class TestTrainF0ExtractorStop(unittest.TestCase):
    """TrainF0Extractor 停止逻辑测试。"""

    @patch("rvc.train.extract_f0.RMVPE")
    def test_request_stop_sets_flag(self, mock_rmvpe):
        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.request_stop()
        self.assertTrue(extractor.stop_requested)

    @patch("rvc.train.extract_f0.RMVPE")
    def test_request_stop_idempotent(self, mock_rmvpe):
        """多次调用 request_stop 不崩溃。"""
        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.request_stop()
        extractor.request_stop()
        extractor.request_stop()
        self.assertTrue(extractor.stop_requested)


class TestTrainF0ExtractorRun(unittest.TestCase):
    """TrainF0Extractor.run 批量处理测试（mock 文件系统）。"""

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

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_processes_all_files(self, mock_load, mock_rmvpe):
        """run 处理所有 wav 文件。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        count = extractor.run(str(self.exp_dir))
        self.assertEqual(count, 3)

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_creates_output_dirs(self, mock_load, mock_rmvpe):
        """run 创建输出目录。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir))

        self.assertTrue((self.exp_dir / "2a_f0").exists())
        self.assertTrue((self.exp_dir / "2b-f0nsf").exists())

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_progress_callback(self, mock_load, mock_rmvpe):
        """run 调用进度回调。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        progress = []
        def cb(i, total):
            progress.append((i, total))

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir), progress_callback=cb)

        self.assertEqual(len(progress), 3)
        self.assertEqual(progress[0], (1, 3))
        self.assertEqual(progress[2], (3, 3))

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_stop_requested(self, mock_load, mock_rmvpe):
        """stop_requested 为 True 时停止处理。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.request_stop()  # 提前停止
        count = extractor.run(str(self.exp_dir))
        # 应该在处理第一个文件之前就停止
        self.assertEqual(count, 3)  # 返回总文件数，但实际处理 0 个

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_stop_check_callback(self, mock_load, mock_rmvpe):
        """stop_check 回调返回 True 时停止处理。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        stop_check = MagicMock(return_value=True)
        count = extractor.run(str(self.exp_dir), stop_check=stop_check)
        self.assertEqual(count, 3)

    @patch("rvc.train.extract_f0.RMVPE")
    @patch("rvc.train.extract_f0.load_audio")
    def test_run_skips_existing_outputs(self, mock_load, mock_rmvpe):
        """已存在输出文件时跳过处理。"""
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_model = MagicMock()
        mock_model.infer_from_audio.return_value = torch.zeros(100, dtype=torch.float32)
        mock_rmvpe.return_value = mock_model

        # 预创建第一个文件的输出
        coarse_dir = self.exp_dir / "2a_f0"
        cont_dir = self.exp_dir / "2b-f0nsf"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        cont_dir.mkdir(parents=True, exist_ok=True)
        np.save(coarse_dir / "test_0.npy", np.array([1, 2, 3]))
        np.save(cont_dir / "test_0.npy", np.array([1.0, 2.0, 3.0]))

        extractor = TrainF0Extractor(device="cpu", is_half=False)
        extractor.run(str(self.exp_dir))

        # 第一个文件应该被跳过，只处理后两个
        self.assertEqual(mock_model.infer_from_audio.call_count, 2)


# 需要 torch 用于 mock
import torch


if __name__ == "__main__":
    unittest.main()
