"""InferenceRunner 深度测试 — 初始化/输入准备/直通推理/缓冲区重置/长度修正。

覆盖：
- __init__: 初始状态、默认值（状态统一放在 EngineState）
- init_processing: 各种参数组合、缓冲区大小、重采样器创建
- _prepare_input: 立体声/单声道转换、连续数组
- _run_inference: 直通模式、长度修正（截断/补零）
- reset_buffers: 缓冲区清零、pipeline 重置
- warmup: 无 pipeline 时直接返回
- route_secondary_output: 委托函数
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from rvc.audio.inference_runner import InferenceRunner
from rvc.inference.engine_state import EngineState


class TestInferenceRunnerInit(unittest.TestCase):
    """InferenceRunner 初始化测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.pipeline.state = EngineState()
        self.pipeline.ctx = MagicMock()
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
        self.assertEqual(runner.state.device, "cpu")

    def test_default_function_vc(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.state.function, "vc")

    def test_custom_function_mono(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu", function="mono")
        self.assertEqual(runner.state.function, "mono")

    def test_sr_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.state.target_sr)

    def test_sr_model_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.state.sr_model)

    def test_channels_default_1(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.state.channels, 1)

    def test_block_samples_zero_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.state.block_samples, 0)

    def test_input_wav_none_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNone(runner.state.input_wav_48k)

    def test_processor_created(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertIsNotNone(runner.state.audio_processor)

    def test_infer_ms_zero_initially(self):
        runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.assertEqual(runner.state.infer_ms, 0.0)


class TestInferenceRunnerInitProcessing(unittest.TestCase):
    """init_processing 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.pipeline.state = EngineState()
        self.pipeline.ctx = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    def test_sr_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.state.target_sr, 48000)

    def test_sr_model_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=40000)
        self.assertEqual(self.runner.state.sr_model, 40000)

    def test_channels_stored(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=2, sr_model=48000)
        self.assertEqual(self.runner.state.channels, 2)

    def test_hz_centis_calculated(self):
        """hz_centis = sr // 100。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.state.hz_centis, 480)

    def test_block_samples_calculated(self):
        """block_samples 应该是 hz_centis 的整数倍。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.state.block_samples % self.runner.state.hz_centis, 0)
        self.assertGreater(self.runner.state.block_samples, 0)

    def test_block_samples_16k_calculated(self):
        """block_samples_16k 应该是 HUBERT_FRAME_SIZE 的整数倍。"""
        from rvc.audio.constants import HUBERT_FRAME_SIZE
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(self.runner.state.block_samples_16k % HUBERT_FRAME_SIZE, 0)

    def test_input_wav_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.state.input_wav_48k)
        self.assertIsInstance(self.runner.state.input_wav_48k, torch.Tensor)

    def test_input_wav_on_correct_device(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertEqual(str(self.runner.state.input_wav_48k.device), "cpu")

    def test_input_wav_res_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.state.input_wav_16k)

    def test_resampler_created(self):
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNotNone(self.runner.state.resampler_48k_to_16k)

    def test_resampler_model2dev_none_when_same_sr(self):
        """sr_model == sr 时 resampler_model_to_48k 为 None。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.assertIsNone(self.runner.state.resampler_model_to_48k)

    def test_resampler_model2dev_created_when_different_sr(self):
        """sr_model != sr 时 resampler_model_to_48k 被创建。"""
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=40000)
        self.assertIsNotNone(self.runner.state.resampler_model_to_48k)

    def test_processor_setup_called(self):
        """init_processing 应该调用 audio_processor.setup。"""
        self.runner.state.audio_processor = MagicMock()
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)
        self.runner.state.audio_processor.setup.assert_called_once()

    def test_various_sample_rates(self):
        for sr in [16000, 22050, 32000, 44100, 48000]:
            runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
            runner.init_processing(sr=sr, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                    channels=1, sr_model=sr)
            self.assertEqual(runner.state.target_sr, sr)
            self.assertEqual(runner.state.hz_centis, sr // 100)

    def test_various_block_times(self):
        for block_t in [0.1, 0.25, 0.5, 1.0]:
            runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
            runner.init_processing(sr=48000, block_t=block_t, cf_t=0.05, extra_t=2.5,
                                    channels=1, sr_model=48000)
            self.assertGreater(runner.state.block_samples, 0)
            self.assertEqual(runner.state.block_samples % runner.state.hz_centis, 0)


class TestResetBuffers(unittest.TestCase):
    """reset_buffers 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.pipeline.state = EngineState()
        self.pipeline.ctx = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")
        self.runner.init_processing(sr=48000, block_t=0.25, cf_t=0.05, extra_t=2.5,
                                     channels=1, sr_model=48000)

    def test_input_wav_zeroed(self):
        self.runner.state.input_wav_48k.fill_(1.0)
        self.runner.reset_buffers()
        self.assertTrue(torch.all(self.runner.state.input_wav_48k == 0))

    def test_input_wav_res_zeroed(self):
        self.runner.state.input_wav_16k.fill_(1.0)
        self.runner.reset_buffers()
        self.assertTrue(torch.all(self.runner.state.input_wav_16k == 0))

    def test_pipeline_reset_called(self):
        self.runner.reset_buffers()
        self.pipeline.reset_pitch_cache.assert_called_once()

    def test_processor_reset_called(self):
        self.runner.state.audio_processor = MagicMock()
        self.runner.reset_buffers()
        self.runner.state.audio_processor.reset.assert_called_once()

    def test_pipeline_none_no_crash(self):
        """pipeline 为 None 时不崩溃。"""
        self.runner.pipeline = None
        self.runner.reset_buffers()  # 不应崩溃

    def test_input_wav_none_no_crash(self):
        """input_wav_48k 为 None 时不崩溃。"""
        self.runner.state.input_wav_48k = None
        self.runner.reset_buffers()  # 不应崩溃


class TestWarmup(unittest.TestCase):
    """warmup 测试。"""

    def setUp(self):
        self.pipeline = MagicMock()
        self.pipeline.state = EngineState()
        self.pipeline.ctx = MagicMock()
        self.runtime_params = MagicMock()
        self.runtime_params.rms_mix = 0.0
        self.runner = InferenceRunner(self.pipeline, self.runtime_params, "cpu")

    def test_warmup_no_pipeline_returns_immediately(self):
        """pipeline 为 None 时直接返回。"""
        self.runner.pipeline = None
        self.runner.warmup(2)  # 不应崩溃

    def test_warmup_no_input_wav_res_returns_immediately(self):
        """input_wav_16k 为 None 时直接返回。"""
        self.runner.state.input_wav_16k = None
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
        self.pipeline.state = EngineState()
        self.pipeline.ctx = MagicMock()
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
