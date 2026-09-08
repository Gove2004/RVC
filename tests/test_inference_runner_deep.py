"""InferenceRunner 深度测试 — 初始化/输入准备/直通推理/缓冲区重置/长度修正。

覆盖：
- __init__: 初始状态、默认值
- init_processing: 各种参数组合、缓冲区大小、重采样器创建
- _prepare_input: 立体声/单声道转换、连续数组
- _run_inference: 直通模式、长度修正（截断/补零）
- reset_buffers: 缓冲区清零、pipeline 重置
- warmup: 无 pipeline 时直接返回
- route_secondary_output: 委托函数
"""
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import torch

from rvc.audio.inference_runner import InferenceRunner


class TestInferenceRunnerInit(unittest.TestCase):
    """InferenceRunner 初始化测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0

    def test_creates_runner(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNotNone(runner)

    def test_pipeline_stored(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIs(runner.pipeline, self.pipeline)

    def test_runtime_params_stored(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIs(runner.runtime_params, self.runtime_params)

    def test_device_stored(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner._device, "cpu")

    def test_default_function_vc(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.function, "vc")

    def test_custom_function_mono(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu", function="mono")
        self.assertEqual(runner.function, "mono")

    def test_sr_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.sr)

    def test_sr_model_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.sr_model)

    def test_channels_default_1(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.channels, 1)

    def test_block_samples_zero_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.block_samples, 0)

    def test_input_wav_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.input_wav)

    def test_processor_created(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNotNone(runner.processor)

    def test_infer_ms_zero_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.infer_ms, 0.0)


class TestInferenceRunnerInitProcessing(unittest.TestCase):
    """init_processing 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    def test_sr_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.sr, 48000)

    def test_sr_model_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=40000)
        self.assertEqual(self.runner.sr_model, 40000)

    def test_channels_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=2, sr_model=48000)
        self.assertEqual(self.runner.channels, 2)

    def test_hz_centis_calculated(self):
        """hz_centis = sr // 100。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.hz_centis, 480)

    def test_block_samples_calculated(self):
        """block_samples 应该是 hz_centis 的整数倍。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.block_samples % self.runner.hz_centis, 0)
        self.assertGreater(self.runner.block_samples, 0)

    def test_block_samples_16k_calculated(self):
        """block_samples_16k 应该是 HUBERT_FRAME_SIZE 的整数倍。"""
        from rvc.audio.constants import HUBERT_FRAME_SIZE
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.block_samples_16k % HUBERT_FRAME_SIZE, 0)

    def test_input_wav_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.input_wav)
        self.assertIsInstance(self.runner.input_wav, torch.Tensor)

    def test_input_wav_on_correct_device(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(str(self.runner.input_wav.device), "cpu")

    def test_input_wav_res_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.input_wav_res)

    def test_resampler_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.resampler)

    def test_resampler_model2dev_none_when_same_sr(self):
        """sr_model == sr 时 resampler_model2dev 为 None。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNone(self.runner.resampler_model2dev)

    def test_resampler_model2dev_created_when_different_sr(self):
        """sr_model != sr 时 resampler_model2dev 被创建。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=40000)
        self.assertIsNotNone(self.runner.resampler_model2dev)

    def test_processor_setup_called(self):
        """init_processing 应该调用 processor.setup。"""
        self.runner.processor = MagicMock()
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.runner.processor.setup.assert_called_once()

    def test_various_sample_rates(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
            runner.init_processing(sr=sr, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                    channels=1, sr_model=sr)
            self.assertEqual(runner.sr, sr)
            self.assertEqual(runner.hz_centis, sr // 100)

    def test_various_block_times(self):
        for block_t in [0.1, 0.25, 0.5, 1.0]:
            runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
            runner.init_processing(sr=48000, block_t=block_t, cf_t=0.05, extra_t=2.5,
                                    channels=1, sr_model=48000)
            self.assertGreater(runner.block_samples, 0)
            self.assertEqual(runner.block_samples % runner.hz_centis, 0)


class TestPrepareInput(unittest.TestCase):
    """_prepare_input 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    def test_mono_input_passthrough(self):
        """单声道输入直接返回（拷贝）。"""
        indata = np.random.randn(1000).astype(np.float32)
        mono = self.runner._prepare_input(indata)
        self.assertEqual(mono.shape, (1000,))
        self.assertTrue(np.allclose(mono, indata))

    def test_stereo_input_averaged(self):
        """立体声输入取左右声道平均值。"""
        indata = np.zeros((1000, 2), dtype=np.float32)
        indata[:, 0] = 1.0
        indata[:, 1] = 0.5
        mono = self.runner._prepare_input(indata)
        self.assertEqual(mono.shape, (1000,))
        self.assertTrue(np.allclose(mono, 0.75))

    def test_multichannel_input_averaged(self):
        """多声道输入取所有声道平均值。"""
        indata = np.ones((1000, 4), dtype=np.float32)
        mono = self.runner._prepare_input(indata)
        self.assertEqual(mono.shape, (1000,))
        self.assertTrue(np.allclose(mono, 1.0))

    def test_output_is_contiguous(self):
        """输出应该是连续数组。"""
        indata = np.random.randn(1000, 2).astype(np.float32)
        mono = self.runner._prepare_input(indata)
        self.assertTrue(mono.flags['C_CONTIGUOUS'])

    def test_output_dtype_float32(self):
        indata = np.random.randn(1000, 2).astype(np.float64)
        mono = self.runner._prepare_input(indata)
        # mean 可能改变 dtype，但应该是浮点
        self.assertTrue(np.issubdtype(mono.dtype, np.floating))

    def test_empty_input(self):
        indata = np.array([], dtype=np.float32)
        mono = self.runner._prepare_input(indata)
        self.assertEqual(mono.shape, (0,))

    def test_single_sample(self):
        indata = np.array([[0.5, 0.3]], dtype=np.float32)
        mono = self.runner._prepare_input(indata)
        self.assertEqual(mono.shape, (1,))
        self.assertAlmostEqual(mono[0], 0.4, places=5)


class TestRunInference(unittest.TestCase):
    """_run_inference 测试（聚焦直通模式和长度修正）。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)

    def test_mono_mode_returns_input_clone(self):
        """function='mono' 时返回输入的克隆（不经过 pipeline）。"""
        self.runner.function = "mono"
        self.runner.input_wav = torch.randn(self.runner.extra_samples + 1000)
        result = self.runner._run_inference()
        expected_len = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        self.assertEqual(result.shape[0], expected_len)

    def test_mono_mode_does_not_call_pipeline(self):
        """function='mono' 时不调用 pipeline.infer。"""
        self.runner.function = "mono"
        self.runner.input_wav = torch.randn(self.runner.extra_samples + 1000)
        self.runner._run_inference()
        self.pipeline.infer.assert_not_called()

    def test_output_length_truncated_when_too_long(self):
        """推理输出过长时被截断到期望长度。"""
        self.runner.function = "mono"
        # 创建一个比期望长的输入
        expected = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        self.runner.input_wav = torch.randn(self.runner.extra_samples + expected + 500)
        result = self.runner._run_inference()
        self.assertEqual(result.shape[0], expected)

    def test_output_length_padded_when_too_short(self):
        """推理输出过短时被补零到期望长度。"""
        self.runner.function = "mono"
        expected = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        # 创建一个比期望短的输入（通过修改 extra_samples）
        self.runner.input_wav = torch.randn(self.runner.extra_samples + expected - 100)
        result = self.runner._run_inference()
        self.assertEqual(result.shape[0], expected)
        # 末尾应该是补的零
        self.assertTrue(torch.all(result[-100:] == 0))

    def test_expected_length_calculation(self):
        """期望长度 = block_samples + sola_buffer_samples + sola_search_samples。"""
        expected = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        self.assertGreater(expected, 0)
        self.assertEqual(expected % self.runner.hz_centis, 0)

    def test_vc_mode_calls_pipeline(self):
        """function='vc' 且有 pipeline 时调用 pipeline.infer。"""
        self.runner.function = "vc"
        self.runner.pipeline = self.pipeline
        # mock pipeline.infer 返回正确长度的张量
        expected = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        self.pipeline.infer.return_value = torch.randn(expected)
        self.runner.input_wav_res = torch.randn(1000)
        result = self.runner._run_inference()
        self.pipeline.infer.assert_called_once()
        self.assertEqual(result.shape[0], expected)

    def test_vc_mode_no_pipeline_returns_mono(self):
        """function='vc' 但 pipeline 为 None 时走直通路径。"""
        self.runner.function = "vc"
        self.runner.pipeline = None
        self.runner.input_wav = torch.randn(self.runner.extra_samples + 1000)
        result = self.runner._run_inference()
        expected = self.runner.block_samples + self.runner.sola_buffer_samples + self.runner.sola_search_samples
        self.assertEqual(result.shape[0], expected)


class TestResetBuffers(unittest.TestCase):
    """reset_buffers 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)

    def test_input_wav_zeroed(self):
        self.runner.input_wav.fill_(1.0)
        self.runner.reset_buffers()
        self.assertTrue(torch.all(self.runner.input_wav == 0))

    def test_input_wav_res_zeroed(self):
        self.runner.input_wav_res.fill_(1.0)
        self.runner.reset_buffers()
        self.assertTrue(torch.all(self.runner.input_wav_res == 0))

    def test_pipeline_reset_called(self):
        self.runner.reset_buffers()
        self.pipeline.reset_pitch_cache.assert_called_once()

    def test_processor_reset_called(self):
        self.runner.processor = MagicMock()
        self.runner.reset_buffers()
        self.runner.processor.reset.assert_called_once()

    def test_pipeline_none_no_crash(self):
        """pipeline 为 None 时不崩溃。"""
        self.runner.pipeline = None
        self.runner.reset_buffers()  # 不应崩溃

    def test_input_wav_none_no_crash(self):
        """input_wav 为 None 时不崩溃。"""
        self.runner.input_wav = None
        self.runner.reset_buffers()  # 不应崩溃


class TestWarmup(unittest.TestCase):
    """warmup 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    def test_warmup_no_pipeline_returns_immediately(self):
        """pipeline 为 None 时直接返回。"""
        self.runner.pipeline = None
        self.runner.warmup(2)  # 不应崩溃

    def test_warmup_no_input_wav_res_returns_immediately(self):
        """input_wav_res 为 None 时直接返回。"""
        self.runner.input_wav_res = None
        self.runner.warmup(2)  # 不应崩溃

    def test_warmup_with_initialized_runner(self):
        """初始化后的 runner 可以 warmup（mock process_block）。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.runner.process_block = MagicMock()
        self.runner.warmup(n=3)
        self.assertEqual(self.runner.process_block.call_count, 3)


class TestRouteSecondaryOutput(unittest.TestCase):
    """route_secondary_output 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    @patch("rvc.audio.inference_runner.route_secondary_output")
    def test_delegates_to_module_function(self, mock_route):
        """route_secondary_output 委托给模块级函数。"""
        outdata = np.zeros((100, 2), dtype=np.float32)
        stream2 = MagicMock()
        out2_q = MagicMock()
        self.runner.route_secondary_output(outdata, stream2, out2_q, True)
        mock_route.assert_called_once_with(outdata, stream2, out2_q, True)


if __name__ == "__main__":
    unittest.main()
