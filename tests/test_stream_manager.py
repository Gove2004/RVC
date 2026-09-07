"""AudioStreamManager 单元测试 — 音频流管理（不依赖实际设备）。"""
import pytest
import numpy as np

from rvc.audio.stream_manager import AudioStreamManager
from rvc.core.errors import AudioDeviceError


class TestAudioStreamManager:
    """AudioStreamManager 基本接口测试。"""

    def test_creation(self):
        mgr = AudioStreamManager()
        assert mgr.stream is None
        assert mgr.stream2 is None
        assert mgr.enable_out2 is False
        assert mgr.running is False
        assert mgr.channels == 1

    def test_stop_all_no_crash(self):
        """无流时 stop_all 不应崩溃。"""
        mgr = AudioStreamManager()
        mgr.stop_all()  # 不应抛异常
        assert mgr.stream is None
        assert mgr.enable_out2 is False

    def test_safe_close_stream_none(self):
        """_safe_close_stream(None) 不应崩溃。"""
        AudioStreamManager._safe_close_stream(None)

    def test_out2_queue_creation(self):
        """副输出队列应在初始化时创建。"""
        mgr = AudioStreamManager()
        assert mgr.out2_q is not None
        assert mgr.out2_q.maxsize == 10

    def test_validate_secondary_output_raises(self):
        """副输出设备通道数不足时应抛 AudioDeviceError。

        注意：此测试需要实际音频设备，跳过。
        """
        pytest.skip("需要实际音频设备")


class TestAudioDeviceError:
    """AudioDeviceError 异常测试。"""

    def test_creation(self):
        err = AudioDeviceError("设备不存在")
        assert str(err) == "设备不存在"
        assert isinstance(err, Exception)

    def test_inheritance(self):
        from rvc.core.errors import RVCError
        assert issubclass(AudioDeviceError, RVCError)
