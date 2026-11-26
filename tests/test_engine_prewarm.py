"""后台预暖（prewarm_runner）的分支与几何复用契约 — 无 GPU。

预暖是纯优化：提前把图捕获到共享缓存的模型上，首次点击「变声」命中同几何即纯
replay。这里锁住几条不能破的边界：
  1. 无 pipeline / 已在运行 / 离线在跑时必须跳过（绝不与 setup/offline 抢 state）；
  2. 跳过时不得释放并非自己持有的 _offline_lock；
  3. 成功预暖留下几何签名，setup 命中同签名时把 warmup 退化为少量 replay。
用 VoiceEngine.__new__ 绕过 __init__ 的 CUDA 探测，mock _make_runner 与 sounddevice。
"""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from rvc.streaming.engine import (
    _PREWARMED_WARMUP_ITERATIONS,
    _WARMUP_ITERATIONS,
    VoiceEngine,
)


def _bare_engine():
    eng = VoiceEngine.__new__(VoiceEngine)
    eng.pipeline = SimpleNamespace(model_sr=40000, hubert_variant="base")
    eng.pth_path = "x.pth"
    eng.running = False
    eng._offline_lock = threading.Lock()
    eng._setup_lock = threading.Lock()
    eng._prewarm_sig = None
    eng._make_runner = mock.Mock()
    return eng


_QUERY = {"default_samplerate": 48000, "max_input_channels": 2, "max_output_channels": 2}


class TestPrewarmSkip(unittest.TestCase):
    def test_skip_when_no_pipeline(self):
        eng = _bare_engine()
        eng.pipeline = None
        with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY):
            eng.prewarm_runner("model", 0, 1, 0.1, 0.1, 2.5)
        eng._make_runner.assert_not_called()
        self.assertIsNone(eng._prewarm_sig)

    def test_skip_when_running(self):
        eng = _bare_engine()
        eng.running = True
        with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY):
            eng.prewarm_runner("model", 0, 1, 0.1, 0.1, 2.5)
        eng._make_runner.assert_not_called()

    def test_skip_when_offline_busy_and_keep_lock(self):
        eng = _bare_engine()
        eng._offline_lock.acquire()  # 模拟离线转换持锁
        try:
            with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY):
                eng.prewarm_runner("model", 0, 1, 0.1, 0.1, 2.5)
            eng._make_runner.assert_not_called()
            # 预暖没拿到锁，绝不能释放别人持有的锁
            self.assertTrue(eng._offline_lock.locked())
        finally:
            eng._offline_lock.release()


class TestPrewarmCapture(unittest.TestCase):
    def test_capture_sets_sig_and_releases_lock(self):
        eng = _bare_engine()
        with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY):
            eng.prewarm_runner("model", 0, 1, 0.1, 0.1, 2.5)
        eng._make_runner.assert_called_once()
        kwargs = eng._make_runner.call_args.kwargs
        self.assertTrue(kwargs["prewarm_formant"])
        self.assertTrue(kwargs["reset_buffers"])
        self.assertEqual(eng._prewarm_sig,
                         eng._runner_sig(40000, 2, 0.1, 0.1, 2.5))
        self.assertFalse(eng._offline_lock.locked())

    def test_failure_is_silent_and_releases_lock(self):
        eng = _bare_engine()
        eng._make_runner.side_effect = RuntimeError("boom")
        with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY):
            eng.prewarm_runner("model", 0, 1, 0.1, 0.1, 2.5)  # 不抛
        self.assertFalse(eng._offline_lock.locked())
        self.assertIsNone(eng._prewarm_sig)


class TestSetupWarmupIters(unittest.TestCase):
    def _setup_engine(self):
        eng = _bare_engine()
        eng._streams = SimpleNamespace(
            main_stream=None, channels=1,
            validate_and_log_devices=mock.Mock(),
            start_main_stream=mock.Mock(),
        )
        eng._saved_default_device = None
        eng.device_sr = 0
        eng.model_sr = 0
        eng.stop = mock.Mock()
        eng.setup_out2 = mock.Mock()
        eng._make_runner.return_value = SimpleNamespace(
            state=SimpleNamespace(block_samples=100))
        return eng

    def _run_setup(self, eng):
        with mock.patch("rvc.streaming.engine.sd.query_devices", return_value=_QUERY), \
             mock.patch("rvc.streaming.engine.sd.default") as default:
            default.device = [0, 1]
            eng.setup("model", 0, 1, 0.1, 0.1, 2.5)
        return eng._make_runner.call_args.kwargs["warmup_iters"]

    def test_full_warmup_without_prewarm(self):
        eng = self._setup_engine()
        self.assertEqual(self._run_setup(eng), _WARMUP_ITERATIONS)

    def test_reduced_warmup_when_sig_matches(self):
        eng = self._setup_engine()
        eng._prewarm_sig = eng._runner_sig(40000, 1, 0.1, 0.1, 2.5)
        self.assertEqual(self._run_setup(eng), _PREWARMED_WARMUP_ITERATIONS)

    def test_full_warmup_when_geometry_differs(self):
        eng = self._setup_engine()
        eng._prewarm_sig = eng._runner_sig(48000, 1, 0.1, 0.1, 2.5)  # 不同 work_sr
        self.assertEqual(self._run_setup(eng), _WARMUP_ITERATIONS)


if __name__ == "__main__":
    unittest.main()