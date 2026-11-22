"""离线文件推理的块循环契约 — 切块对齐、pad 裁剪、取消语义。

这段逻辑从 VoiceEngine 拆到 rvc.streaming.offline 后单独可测：
不需要 GPU、不需要设备流，用假 runner 就能锁住「输入多长、输出多长」。
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rvc.core.errors import CancelledError
from rvc.streaming import offline


class _EchoRunner:
    """假 runner：process_block 把输入原样写进 outdata，并记录块数。

    state 延迟置零：回声直通无流式延迟，裁剪退化为 rendered[pad:pad+len]。
    """

    def __init__(self):
        self.calls = 0
        self.state = SimpleNamespace(crossfade_samples=0, sola_search_samples=0)

    def process_block(self, indata, outdata, frames):
        self.calls += 1
        outdata[:, 0] = indata[:frames]


class TestRunStream(unittest.TestCase):
    def test_output_length_matches_input(self):
        wav = np.arange(5000, dtype=np.float32)
        out = offline.run_stream(wav, _EchoRunner(), block=1000, pad=300)
        self.assertEqual(out.shape, wav.shape)

    def test_pad_is_trimmed_from_timeline(self):
        """两端 pad 只是给 SOLA/上下文留边，输出必须与原文件逐样本对齐。"""
        wav = np.arange(10, dtype=np.float32)
        out = offline.run_stream(wav, _EchoRunner(), block=4, pad=3)
        np.testing.assert_allclose(out, wav)

    def test_crop_compensates_streaming_delay(self):
        """合成窗比当前块早 crossfade+search，裁剪须补掉这段延迟，否则输出后移。"""
        crossfade, search = 40, 10
        runner = _EchoRunner()
        runner.state = SimpleNamespace(crossfade_samples=crossfade,
                                       sola_search_samples=search)
        wav = np.arange(2000, dtype=np.float32)
        block, pad = 100, 300
        out = offline.run_stream(wav, runner, block=block, pad=pad)
        # 回声直通：rendered == padded（块对齐）；裁剪应取 [pad+delay : pad+delay+len]
        padded = np.pad(wav, (pad, pad), mode="reflect")
        delay = crossfade + search
        expected = padded[pad + delay: pad + delay + len(wav)]
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, wav.shape)

    def test_blocks_are_padded_to_whole_count(self):
        """pad 后不足整块要补零，否则尾部样本会被丢掉。"""
        runner = _EchoRunner()
        wav = np.zeros(1000, dtype=np.float32)
        offline.run_stream(wav, runner, block=600, pad=200)
        # 1000 + 400 = 1400 → 向上取整到 3 块
        self.assertEqual(runner.calls, 3)

    def test_short_audio_uses_constant_pad(self):
        """len(wav) <= pad 时 np.pad 的 reflect 会抛错，必须退回 constant。"""
        out = offline.run_stream(np.ones(5, dtype=np.float32), _EchoRunner(),
                                 block=100, pad=300)
        self.assertEqual(out.shape, (5,))

    def test_progress_reports_monotonic_counts(self):
        seen = []
        offline.run_stream(np.zeros(2400, dtype=np.float32), _EchoRunner(),
                           block=600, pad=0,
                           progress_cb=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 4), (2, 4), (3, 4), (4, 4)])

    def test_cancel_raises_before_first_block(self):
        runner = _EchoRunner()
        with self.assertRaises(CancelledError):
            offline.run_stream(np.zeros(1200, dtype=np.float32), runner,
                               block=600, pad=0, cancel_cb=lambda: True)
        self.assertEqual(runner.calls, 0)

    def test_cancel_midway_stops_consuming(self):
        runner = _EchoRunner()
        state = {"blocks": 0}

        def cancel():
            state["blocks"] += 1
            return state["blocks"] > 2

        with self.assertRaises(CancelledError):
            offline.run_stream(np.zeros(6000, dtype=np.float32), runner,
                               block=600, pad=0, cancel_cb=cancel)
        self.assertEqual(runner.calls, 2, "取消后不应再跑第 3 块")

    def test_write_output_uses_float_subtype(self):
        with mock.patch("rvc.io.wav_file.write_wav") as writer:
            offline.write_output(np.zeros(10, dtype=np.float32), "out.wav", 40000)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["subtype"], "FLOAT")


class TestProcessFileNotReentrant(unittest.TestCase):
    """process_file 必须不可重入：并发调用要立刻报可读错，而非静默产生 CUDA 错。

    背景：process_file 会替换 engine._runner 并原地改 runtime_params，两个并发
    调用会互相踩——实测两线程同时进来时 CUDA Graph 捕获被对方破坏，抛
    AcceleratorError，且**两路都没产出文件、只有一路报错**，排查成本很高。

    GUI 侧有三道守卫（_converting / is_running / is_engine_loading）当前不可达，
    这里兜的是非 GUI 调用方与将来守卫被改的情况。
    """

    def _engine(self):
        from rvc.streaming.engine import VoiceEngine

        e = VoiceEngine.__new__(VoiceEngine)   # 绕过 __init__（要真 CUDA）
        import threading
        e._offline_lock = threading.Lock()
        e.pipeline = SimpleNamespace(model_sr=40000)
        return e

    def test_second_call_raises_readable_error(self):
        import threading
        import time

        e = self._engine()
        released = threading.Event()

        def holder():
            with e._offline_lock:
                released.wait(3.0)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        time.sleep(0.15)          # 确保 holder 已持锁
        try:
            with self.assertRaises(RuntimeError) as ctx:
                e.process_file(SimpleNamespace(input_path="x.wav",
                                               output_path="y.wav"))
            msg = str(ctx.exception)
            self.assertIn("不可重入", msg)
            self.assertIn("离线转换正在进行", msg)
        finally:
            released.set()
            t.join(3.0)

    def test_lock_released_after_failure(self):
        """异常路径也必须释放锁，否则引擎永久卡死。"""
        e = self._engine()
        # 用一个不存在的输入路径触发 load_audio_at_sr 抛错
        with self.assertRaises(Exception):
            e.process_file(SimpleNamespace(input_path="E:/__no_such__.wav",
                                           output_path="E:/__no_such_out__.wav"))
        # 锁应已释放
        self.assertTrue(e._offline_lock.acquire(blocking=False),
                        "process_file 抛错后没释放 _offline_lock")
        e._offline_lock.release()


if __name__ == "__main__":
    unittest.main()
