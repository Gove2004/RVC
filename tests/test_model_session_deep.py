"""模型会话深度测试 — ModelSession dataclass / _DeviceConfig / Manager 纯逻辑。

覆盖：
- ModelSession: dataclass 初始化、字段、默认值
- _DeviceConfig: 初始化、字段
- ModelSessionManager: 初始化、current_session 属性、clear_all 无 session
- 注意：load/get_f0_extractor 需要真实模型，只测试不调用真实模型的部分
"""
import unittest
from unittest.mock import MagicMock, patch

from rvc.inference.model_session import (
    ModelSession,
    ModelSessionManager,
    _DeviceConfig,
)


class TestModelSession(unittest.TestCase):
    """ModelSession dataclass 测试。"""

    def test_create_with_all_fields(self):
        session = ModelSession(
            hubert="hubert_obj",
            synthesizer="syn_obj",
            target_sr=48000,
            use_f0=1,
        )
        self.assertEqual(session.hubert, "hubert_obj")
        self.assertEqual(session.synthesizer, "syn_obj")
        self.assertEqual(session.target_sr, 48000)
        self.assertEqual(session.use_f0, 1)

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        self.assertTrue(is_dataclass(ModelSession))

    def test_fields_exist(self):
        session = ModelSession(
            hubert=None, synthesizer=None, target_sr=0, use_f0=0,
        )
        self.assertTrue(hasattr(session, "hubert"))
        self.assertTrue(hasattr(session, "synthesizer"))
        self.assertTrue(hasattr(session, "target_sr"))
        self.assertTrue(hasattr(session, "use_f0"))

    def test_various_target_sr(self):
        for sr in [16000, 22050, 32000, 40000, 44100, 48000]:
            session = ModelSession(
                hubert=None, synthesizer=None, target_sr=sr, use_f0=1,
            )
            self.assertEqual(session.target_sr, sr)

    def test_use_f0_values(self):
        for use_f0 in [0, 1]:
            session = ModelSession(
                hubert=None, synthesizer=None, target_sr=48000, use_f0=use_f0,
            )
            self.assertEqual(session.use_f0, use_f0)

    def test_mutable_fields(self):
        """dataclass 字段是可变的。"""
        session = ModelSession(
            hubert="old", synthesizer="old", target_sr=48000, use_f0=1,
        )
        session.hubert = "new"
        session.target_sr = 40000
        self.assertEqual(session.hubert, "new")
        self.assertEqual(session.target_sr, 40000)


class TestDeviceConfig(unittest.TestCase):
    """_DeviceConfig 测试。"""

    def test_create(self):
        config = _DeviceConfig(device="cuda:0", is_half=True)
        self.assertEqual(config.device, "cuda:0")
        self.assertTrue(config.is_half)

    def test_cpu_device(self):
        config = _DeviceConfig(device="cpu", is_half=False)
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.is_half)

    def test_various_devices(self):
        for device in ["cpu", "cuda:0", "cuda:1", "mps"]:
            config = _DeviceConfig(device=device, is_half=False)
            self.assertEqual(config.device, device)

    def test_is_half_true(self):
        config = _DeviceConfig(device="cuda:0", is_half=True)
        self.assertTrue(config.is_half)

    def test_is_half_false(self):
        config = _DeviceConfig(device="cuda:0", is_half=False)
        self.assertFalse(config.is_half)

    def test_fields_mutable(self):
        config = _DeviceConfig(device="cpu", is_half=False)
        config.device = "cuda:0"
        config.is_half = True
        self.assertEqual(config.device, "cuda:0")
        self.assertTrue(config.is_half)


class TestModelSessionManagerInit(unittest.TestCase):
    """ModelSessionManager 初始化测试。"""

    def setUp(self):
        self.device_config = MagicMock()
        self.device_config.device = "cpu"
        self.device_config.is_half = False

    def test_creates_manager(self):
        manager = ModelSessionManager(self.device_config)
        self.assertIsNotNone(manager)

    def test_device_stored(self):
        manager = ModelSessionManager(self.device_config)
        self.assertEqual(manager.device, "cpu")

    def test_is_half_stored(self):
        manager = ModelSessionManager(self.device_config)
        self.assertFalse(manager.is_half)

    def test_current_session_none_initially(self):
        manager = ModelSessionManager(self.device_config)
        self.assertIsNone(manager.current_session)

    def test_current_pth_none_initially(self):
        manager = ModelSessionManager(self.device_config)
        self.assertIsNone(manager._current_pth)

    def test_inference_cache_default(self):
        """inference_cache=None 时使用默认全局缓存。"""
        manager = ModelSessionManager(self.device_config)
        self.assertIsNotNone(manager.inference_cache)

    def test_inference_cache_custom(self):
        custom_cache = MagicMock()
        manager = ModelSessionManager(self.device_config, inference_cache=custom_cache)
        self.assertIs(manager.inference_cache, custom_cache)

    def test_cuda_device(self):
        device_config = MagicMock()
        device_config.device = "cuda:0"
        device_config.is_half = True
        manager = ModelSessionManager(device_config)
        self.assertEqual(manager.device, "cuda:0")
        self.assertTrue(manager.is_half)


class TestModelSessionManagerClearAll(unittest.TestCase):
    """ModelSessionManager.clear_all 测试（无真实模型）。"""

    def setUp(self):
        self.device_config = MagicMock()
        self.device_config.device = "cpu"
        self.device_config.is_half = False
        self.inference_cache = MagicMock()
        # _synthesizer.values() 返回空列表
        self.inference_cache._synthesizer.values.return_value = []

    def test_clear_all_no_session_no_crash(self):
        """没有当前 session 时 clear_all 不崩溃。"""
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        manager.clear_all()  # 不应崩溃

    def test_clear_all_resets_current_pth(self):
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        manager._current_pth = "/path/model.pth"
        manager.clear_all()
        self.assertIsNone(manager._current_pth)

    def test_clear_all_resets_current_session(self):
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        manager._current_session = MagicMock()
        manager.clear_all()
        self.assertIsNone(manager._current_session)

    def test_clear_all_calls_clear_f0_caches(self):
        """clear_all 调用 inference_cache.clear_f0_cuda_graph_caches。"""
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        manager.clear_all()
        self.inference_cache.clear_f0_cuda_graph_caches.assert_called_once()

    def test_clear_all_with_session_clears_cuda_graph(self):
        """有当前 session 时清除 synthesizer 和 hubert 的 CUDA Graph。"""
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        session = MagicMock()
        manager._current_session = session
        with patch("rvc.inference.model_session.clear_cuda_graph_cache") as mock_clear:
            manager.clear_all()
            # 应该调用 2 次：synthesizer 和 hubert
            self.assertEqual(mock_clear.call_count, 2)

    def test_clear_all_iterates_synthesizer_cache(self):
        """clear_all 遍历 inference_cache._synthesizer。"""
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        manager.clear_all()
        self.inference_cache._synthesizer.values.assert_called_once()

    def test_clear_all_with_synthesizer_bundle(self):
        """缓存中有 synthesizer bundle 时清除其 CUDA Graph。"""
        bundle = MagicMock()
        bundle.synthesizer = MagicMock()
        self.inference_cache._synthesizer.values.return_value = [bundle]
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        with patch("rvc.inference.model_session.clear_cuda_graph_cache") as mock_clear:
            manager.clear_all()
            # 应该调用 1 次（bundle 的 synthesizer）
            # 注意：没有当前 session，所以只有 bundle 的
            mock_clear.assert_any_call(bundle.synthesizer)

    def test_clear_all_bundle_without_synthesizer_attr(self):
        """bundle 没有 synthesizer 属性时跳过（hasattr 检查）。"""
        bundle = object()  # 没有 synthesizer 属性
        self.inference_cache._synthesizer.values.return_value = [bundle]
        manager = ModelSessionManager(self.device_config, inference_cache=self.inference_cache)
        with patch("rvc.inference.model_session.clear_cuda_graph_cache") as mock_clear:
            manager.clear_all()  # 不应崩溃
            # 没有当前 session，bundle 也没有 synthesizer 属性，所以不调用
            self.assertEqual(mock_clear.call_count, 0)


class TestModelSessionManagerCurrentSession(unittest.TestCase):
    """current_session 属性测试。"""

    def setUp(self):
        self.device_config = MagicMock()
        self.device_config.device = "cpu"
        self.device_config.is_half = False

    def test_returns_none_when_not_loaded(self):
        manager = ModelSessionManager(self.device_config)
        self.assertIsNone(manager.current_session)

    def test_returns_session_when_loaded(self):
        manager = ModelSessionManager(self.device_config)
        session = ModelSession(
            hubert="h", synthesizer="s", target_sr=48000, use_f0=1,
        )
        manager._current_session = session
        self.assertIs(manager.current_session, session)

    def test_property_read_only(self):
        """current_session 是只读属性。"""
        manager = ModelSessionManager(self.device_config)
        with self.assertRaises(AttributeError):
            manager.current_session = MagicMock()


if __name__ == "__main__":
    unittest.main()
