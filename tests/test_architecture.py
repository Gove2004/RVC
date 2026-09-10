"""架构验证测试 — 验证重构后的模块导入与接口一致性。"""

import unittest


class TestCoreModuleImports(unittest.TestCase):
    """核心模块导入测试。"""

    def test_config_import(self):
        from rvc.core.config import (
            InferenceConfig, OfflineConfig, TrainConfig, HUBERT_DEFAULT,
        )
        assert InferenceConfig is not None
        assert OfflineConfig is not None
        assert TrainConfig is not None

    def test_errors_import(self):
        from rvc.core.errors import (
            RVCError, ModelLoadError, AudioDeviceError, InferenceError,
            F0ExtractionError, FeatureExtractionError, ConfigError, AudioLoadError,
        )
        assert RVCError is not None
        assert issubclass(ModelLoadError, RVCError)
        assert issubclass(AudioDeviceError, RVCError)
        assert issubclass(F0ExtractionError, InferenceError)

    def test_cuda_graph_import(self):
        from rvc.inference.cuda_graph import (
            run_cuda_graph, clear_cuda_graph_cache, cuda_graph_enabled,
        )
        assert run_cuda_graph is not None


class TestInferenceModuleImports(unittest.TestCase):
    """推理模块导入测试。"""

    def test_model_session_import(self):
        from rvc.inference.model_session import (
            ModelSession, ModelSessionManager,
        )
        assert ModelSession is not None
        assert ModelSessionManager is not None

    def test_f0_extractor_import(self):
        from rvc.inference.f0_extractor import (
            F0Extractor, RMVPEExtractor, FCPEExtractor, create_f0_extractor,
            postprocess_f0,
        )
        assert F0Extractor is not None
        assert issubclass(RMVPEExtractor, F0Extractor)
        assert issubclass(FCPEExtractor, F0Extractor)

    def test_pipeline_import(self):
        from rvc.inference.pipeline import InferencePipeline
        assert InferencePipeline is not None

    def test_synthesis_import(self):
        from rvc.inference.synthesis import infer_synth_audio, apply_formant_resample
        assert infer_synth_audio is not None


class TestAudioModuleImports(unittest.TestCase):
    """音频模块导入测试。"""

    def test_stream_manager_import(self):
        from rvc.audio.stream_manager import AudioStreamManager
        assert AudioStreamManager is not None

    def test_inference_runner_import(self):
        from rvc.audio.inference_runner import InferenceRunner
        assert InferenceRunner is not None

    def test_realtime_engine_import(self):
        from rvc.audio.realtime_engine import RealtimeEngine
        assert RealtimeEngine is not None

    def test_effects_import(self):
        from rvc.audio.effects import AudioProcessor
        assert AudioProcessor is not None


class TestConfigDefaults(unittest.TestCase):
    """配置默认值一致性测试。"""

    def test_inference_config_defaults(self):
        from rvc.core.config import InferenceConfig
        cfg = InferenceConfig()
        assert cfg.formant == 0.0
        assert cfg.protect == 0.5
        assert cfg.f0_method == "rmvpe"
        assert cfg.rms_mix == 0.0

    def test_engine_config_defaults(self):
        from rvc.core.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.block_time == 0.25
        assert cfg.crossfade_time == 0.05
        assert cfg.extra_time == 2.5
        assert cfg.enable_out2 is False

    def test_offline_config_inheritance(self):
        from rvc.core.config import OfflineConfig, InferenceConfig
        cfg = OfflineConfig()
        assert isinstance(cfg, InferenceConfig)
        assert cfg.input_path == ""
        assert cfg.output_path == ""
        assert cfg.model_path == ""

    def test_train_config_defaults(self):
        from rvc.core.config import TrainConfig
        cfg = TrainConfig(exp_dir="/tmp/test")
        assert cfg.exp_dir == "/tmp/test"
        assert cfg.sr == 48000
        assert cfg.epochs == 2000
        assert cfg.batch_size == 4
        assert cfg.fp16_run is True
        assert cfg.keep_ckpts == 1


class TestF0ExtractorInterface(unittest.TestCase):
    """F0 提取器接口一致性测试。"""

    def test_rmvpe_has_clear_cuda_graph(self):
        from rvc.inference.f0_extractor import RMVPEExtractor
        assert hasattr(RMVPEExtractor, "clear_cuda_graph")
        assert callable(getattr(RMVPEExtractor, "clear_cuda_graph"))

    def test_fcpe_has_clear_cuda_graph(self):
        from rvc.inference.f0_extractor import FCPEExtractor
        assert hasattr(FCPEExtractor, "clear_cuda_graph")
        assert callable(getattr(FCPEExtractor, "clear_cuda_graph"))

    def test_abc_has_clear_cuda_graph(self):
        from rvc.inference.f0_extractor import F0Extractor
        assert "clear_cuda_graph" in F0Extractor.__abstractmethods__
