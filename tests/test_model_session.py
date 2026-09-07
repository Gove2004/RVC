"""ModelSessionManager 单元测试 — 模型生命周期管理。"""

import unittest

from rvc.inference.model_session import ModelSession, ModelSessionManager, _DeviceConfig


class TestDeviceConfig(unittest.TestCase):
    """_DeviceConfig 轻量级适配器测试。"""

    def test_attributes(self):
        cfg = _DeviceConfig("cuda:0", True)
        assert cfg.device == "cuda:0"
        assert cfg.is_half is True

    def test_cpu(self):
        cfg = _DeviceConfig("cpu", False)
        assert cfg.device == "cpu"
        assert cfg.is_half is False


class TestModelSession(unittest.TestCase):
    """ModelSession dataclass 测试。"""

    def test_creation(self):
        session = ModelSession(
            hubert="hubert_model",
            synthesizer="syn_model",
            target_sr=40000,
            use_f0=1,
        )
        assert session.hubert == "hubert_model"
        assert session.synthesizer == "syn_model"
        assert session.target_sr == 40000
        assert session.use_f0 == 1


class TestModelSessionManager(unittest.TestCase):
    """ModelSessionManager 基本接口测试（不实际加载模型）。"""

    def test_creation(self):
        cfg = _DeviceConfig("cpu", False)
        manager = ModelSessionManager(cfg)
        assert manager.device == "cpu"
        assert manager.is_half is False
        assert manager.current_session is None
        assert manager._current_pth is None

    def test_clear_all_no_crash(self):
        """无模型时 clear_all 不应崩溃。"""
        cfg = _DeviceConfig("cpu", False)
        manager = ModelSessionManager(cfg)
        manager.clear_all()  # 不应抛异常
        assert manager.current_session is None

    def test_inference_cache_default(self):
        """未传 inference_cache 时应使用默认全局缓存。"""
        from rvc.models.inference_cache import default_inference_cache
        cfg = _DeviceConfig("cpu", False)
        manager = ModelSessionManager(cfg)
        assert manager.inference_cache is default_inference_cache
