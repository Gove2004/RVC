"""RealtimeEngine 深度测试 — 初始化/停止/错误处理/输出归一化/副输出前置条件。

覆盖：
- __init__: 初始状态、默认值、子组件创建
- stop: 停止引擎、状态重置、流关闭
- _cb: 正常回调、错误计数、连续错误停止、副输出路由
- _write_output_wav: 峰值归一化、防削波
- setup_out2: 前置条件检查（主引擎未启动时抛异常）
- 错误统计: error_count/max_error_count/last_error
"""
import unittest
from unittest.mock import MagicMock, patch, call, PropertyMock

import numpy as np
import torch

from rvc.audio.realtime_engine import RealtimeEngine


class TestRealtimeEngineInit(unittest.TestCase):
    """RealtimeEngine 初始化测试。"""

    def setUp(self):
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0

    def test_creates_engine(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIsNotNone(engine)

    def test_runtime_params_stored(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIs(engine.runtime_params, self.runtime_params)

    def test_inference_cache_stored(self):
        cache = MagicMock()
        engine = RealtimeEngine(self.runtime_params, inference_cache=cache)
        self.assertIs(engine.inference_cache, cache)

    def test_inference_cache_none_default(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIsNone(engine.inference_cache)

    def test_on_runtime_error_stored(self):
        cb = MagicMock()
        engine = RealtimeEngine(self.runtime_params, on_runtime_error=cb)
        self.assertIs(engine.on_runtime_error, cb)

    def test_pipeline_none_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIsNone(engine.pipeline)

    def test_running_false_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertFalse(engine.running)

    def test_function_vc_default(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.function, "vc")

    def test_stream_mgr_created(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIsNotNone(engine._stream_mgr)

    def test_runner_none_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertIsNone(engine._runner)

    def test_error_count_zero_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.error_count, 0)

    def test_last_error_empty_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.last_error, "")

    def test_runtime_error_pending_false_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertFalse(engine.runtime_error_pending)

    def test_infer_ms_zero_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.infer_ms, 0.0)

    def test_measure_ms_zero_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.measure_ms, 0.0)

    def test_pth_path_empty_initially(self):
        engine = RealtimeEngine(self.runtime_params)
        self.assertEqual(engine.pth_path, "")


class TestRealtimeEngineStop(unittest.TestCase):
    """stop 测试。"""

    def setUp(self):
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.engine = RealtimeEngine(self.runtime_params)
        self.engine._runner = MagicMock()

    def test_stop_sets_running_false(self):
        self.engine.running = True
        self.engine.stop()
        self.assertFalse(self.engine.running)

    def test_stop_resets_error_count(self):
        self.engine._runner.error_count = 5
        self.engine.stop()
        self.engine._runner.reset_error_state.assert_called_once()
    def test_stop_resets_runtime_error_pending(self):
        self.engine._runner.runtime_error_pending = True
        self.engine.stop()
        self.engine._runner.reset_error_state.assert_called_once()

    def test_stop_calls_stream_mgr_stop_all(self):
        self.engine._stream_mgr = MagicMock()
        self.engine.stop()
        self.engine._stream_mgr.stop_all.assert_called_once()

    def test_stop_no_crash_when_not_running(self):
        """未运行时调用 stop 不应崩溃。"""
        self.engine.stop()  # 不应崩溃

    @patch("rvc.audio.realtime_engine.torch")
    def test_stop_cuda_synchronize_when_available(self, mock_torch):
        """CUDA 可用时调用 synchronize。"""
        mock_torch.cuda.is_available.return_value = True
        self.engine.stop()
        mock_torch.cuda.synchronize.assert_called_once()

    @patch("rvc.audio.realtime_engine.torch")
    def test_stop_no_synchronize_when_cuda_unavailable(self, mock_torch):
        """CUDA 不可用时不调用 synchronize。"""
        mock_torch.cuda.is_available.return_value = False
        self.engine.stop()
        mock_torch.cuda.synchronize.assert_not_called()


class TestRealtimeEngineCallback(unittest.TestCase):
    """_cb 回调函数测试。"""

    def setUp(self):
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.engine = RealtimeEngine(self.runtime_params)
        self.engine._runner = MagicMock()
        self.engine._runner.infer_ms = 5.0
        self.engine._stream_mgr = MagicMock()
        self.engine._stream_mgr.enable_out2 = False

        # 创建模拟的 times 对象
        self.times = MagicMock()
        self.times.outputBufferDacTime = 1.0
        self.times.inputBufferAdcTime = 0.99

    def test_callback_calls_runner_process_block(self):
        """正常回调调用 runner.process_block。"""
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine._runner.process_block.assert_called_once()

    def test_callback_updates_infer_ms(self):
        """回调后更新 infer_ms。"""
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertEqual(self.engine.infer_ms, 5.0)

    def test_callback_resets_error_count_on_success(self):
        self.engine._runner.error_count = 2
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertEqual(self.engine._runner.error_count, 0)
    def test_callback_error_increments_count(self):
        self.engine._runner.process_block.side_effect = RuntimeError("test error")
        self.engine._runner.handle_error.return_value = False
        self.engine._runner.error_count = 0
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine._runner.handle_error.assert_called_once()
    def test_callback_error_sets_last_error(self):
        self.engine._runner.process_block.side_effect = RuntimeError("test error message")
        self.engine._runner.handle_error.return_value = False
        self.engine._runner.last_error = "test error message"
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertIn("test error message", self.engine.last_error)
    def test_callback_error_zeros_outdata(self):
        self.engine._runner.process_block.side_effect = RuntimeError("test")
        self.engine._runner.handle_error.return_value = False
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.ones((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertTrue(np.all(outdata == 0))
    def test_callback_stops_after_max_errors(self):
        self.engine._runner.process_block.side_effect = RuntimeError("test")
        self.engine.on_runtime_error = MagicMock()
        self.engine.running = True
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)

        # 前两次错误不停止
        self.engine._runner.handle_error.return_value = False
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertTrue(self.engine.running)

        # 第三次错误触发停止
        self.engine._runner.handle_error.return_value = True
        with self.assertRaises(Exception):
            self.engine._cb(indata, outdata, 100, self.times, None)
        self.assertFalse(self.engine.running)
        self.engine.on_runtime_error.assert_called_once()
    def test_callback_calls_on_runtime_error(self):
        self.engine._runner.process_block.side_effect = RuntimeError("test error")
        self.engine._runner.handle_error.return_value = True
        self.engine.on_runtime_error = MagicMock()
        self.engine.running = True
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)

        with self.assertRaises(Exception):
            self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine.on_runtime_error.assert_called_once()
    def test_callback_secondary_output_routed_when_enabled(self):
        """副输出启用时调用 route_secondary_output。"""
        self.engine._stream_mgr.enable_out2 = True
        self.engine._stream_mgr.stream2 = MagicMock()
        self.engine._stream_mgr.out2_q = MagicMock()
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine._runner.route_secondary_output.assert_called_once()

    def test_callback_no_secondary_output_when_disabled(self):
        """副输出禁用时不调用 route_secondary_output。"""
        self.engine._stream_mgr.enable_out2 = False
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        self.engine._runner.route_secondary_output.assert_not_called()

    def test_callback_measure_ms_updated(self):
        """回调后更新 measure_ms（瞬时值）。"""
        self.engine.measure_ms = 0.0
        indata = np.zeros((100, 1), dtype=np.float32)
        outdata = np.zeros((100, 1), dtype=np.float32)
        self.engine._cb(indata, outdata, 100, self.times, None)
        # d = 1.0 - 0.99 = 0.01s = 10ms
        self.assertGreater(self.engine.measure_ms, 0)

class TestWriteOutputWav(unittest.TestCase):
    """_write_output_wav 测试（峰值归一化）。"""

    def setUp(self):
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.engine = RealtimeEngine(self.runtime_params)

    @patch("rvc.audio.wav_io.write_wav")
    def test_no_normalization_when_below_threshold(self, mock_write):
        """峰值低于 0.99 时不归一化。"""
        result = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        self.engine._write_output_wav(result, "/tmp/test.wav", 48000)
        # 写入的数据应该和原始数据相同
        written_data = mock_write.call_args[0][1]
        self.assertTrue(np.allclose(written_data, result))

    @patch("rvc.audio.wav_io.write_wav")
    def test_normalization_when_above_threshold(self, mock_write):
        """峰值高于 0.99 时归一化（防削波）。"""
        result = np.array([1.5, -1.2, 0.8], dtype=np.float32)
        self.engine._write_output_wav(result, "/tmp/test.wav", 48000)
        written_data = mock_write.call_args[0][1]
        # 峰值应该被归一化到 0.99
        self.assertAlmostEqual(np.abs(written_data).max(), 0.99, places=3)

    @patch("rvc.audio.wav_io.write_wav")
    def test_normalization_preserves_phase(self, mock_write):
        """归一化保留相位（正负号）。"""
        result = np.array([1.5, -1.2, 0.8], dtype=np.float32)
        self.engine._write_output_wav(result, "/tmp/test.wav", 48000)
        written_data = mock_write.call_args[0][1]
        self.assertTrue((np.sign(written_data) == np.sign(result)).all())

    @patch("rvc.audio.wav_io.write_wav")
    def test_write_wav_called_with_correct_params(self, mock_write):
        """write_wav 被正确调用。"""
        result = np.array([0.5], dtype=np.float32)
        self.engine._write_output_wav(result, "/tmp/test.wav", 48000)
        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        self.assertEqual(args[0], "/tmp/test.wav")
        self.assertEqual(args[2], 48000)
        self.assertEqual(mock_write.call_args[1]["subtype"], "FLOAT")

    @patch("rvc.audio.wav_io.write_wav")
    def test_empty_array_no_crash(self, mock_write):
        """空数组不崩溃。"""
        result = np.array([], dtype=np.float32)
        self.engine._write_output_wav(result, "/tmp/test.wav", 48000)
        mock_write.assert_called_once()


class TestSetupOut2(unittest.TestCase):
    """setup_out2 测试。"""

    def setUp(self):
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.engine = RealtimeEngine(self.runtime_params)

    def test_setup_out2_raises_when_runner_none(self):
        """主引擎未启动时 setup_out2 抛异常。"""
        self.engine._runner = None
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.setup_out2(1)
        self.assertIn("先启动主引擎", str(ctx.exception))

    def test_setup_out2_calls_stream_mgr(self):
        """主引擎已启动时调用 stream_mgr.start_secondary_output。"""
        self.engine._runner = MagicMock()
        self.engine._runner.sr = 48000
        self.engine._runner.channels = 2
        self.engine._runner.block_samples = 12000
        self.engine._stream_mgr = MagicMock()
        self.engine.setup_out2(3)
        self.engine._stream_mgr.start_secondary_output.assert_called_once_with(
            3, 48000, 2, 12000
        )


if __name__ == "__main__":
    unittest.main()
