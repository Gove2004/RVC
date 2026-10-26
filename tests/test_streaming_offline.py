"""离线文件推理的块循环契约 — 切块对齐、pad 裁剪、取消语义。

这段逻辑从 VoiceEngine 拆到 rvc.streaming.offline 后单独可测：
不需要 GPU、不需要设备流，用假 runner 就能锁住「输入多长、输出多长」。
"""
import unittest
from unittest import mock

import numpy as np

from rvc.core.errors import CancelledError
from rvc.streaming import offline


class _EchoRunner:
    """假 runner：process_block 把输入原样写进 outdata，并记录块数。"""

    def __init__(self):
        self.calls = 0

    def process_block(self, indata, outdata, frames):
        self.calls += 1
        outdata[:, 0] = indata[:frames]


class _DelayedEchoRunner(_EchoRunner):
    """模拟带固定未来上下文的 runner，离线端应移除该时间偏移。"""

    def __init__(self, lookahead_samples):
        super().__init__()
        self.state = type("State", (), {"lookahead_samples": lookahead_samples})()
        self._queue = np.zeros(lookahead_samples, dtype=np.float32)

    def process_block(self, indata, outdata, frames):
        self.calls += 1
        joined = np.concatenate((self._queue, np.asarray(indata[:frames])))
        outdata[:, 0] = joined[:frames]
        self._queue = joined[frames:]


class TestRunStream(unittest.TestCase):
    def test_output_length_matches_input(self):
        wav = np.arange(5000, dtype=np.float32)
        out = offline.run_stream(wav, _EchoRunner(), block=1000, pad=300)
        self.assertEqual(out.shape, wav.shape)

    def test_lookahead_is_trimmed_from_offline_timeline(self):
        wav = np.arange(10, dtype=np.float32)
        out = offline.run_stream(wav, _DelayedEchoRunner(2), block=4, pad=3)
        self.assertEqual(out.shape, wav.shape)
        np.testing.assert_allclose(out, wav)

    def test_small_pad_still_returns_full_lookahead_output(self):
        wav = np.arange(10, dtype=np.float32)
        runner = _DelayedEchoRunner(2)
        out = offline.run_stream(wav, runner, block=4, pad=0)
        self.assertEqual(out.shape, wav.shape)
        np.testing.assert_allclose(out, wav)

    def test_large_pad_does_not_add_redundant_tail_blocks(self):
        runner = _DelayedEchoRunner(2)
        offline.run_stream(np.arange(10, dtype=np.float32), runner, block=4, pad=3)
        # 10 + 2*3 = 16 samples already cover the 2-sample lookahead; no extra block.
        self.assertEqual(runner.calls, 4)

    def test_invalid_lookahead_metadata_falls_back_to_zero(self):
        runner = _EchoRunner()
        runner.state = type("State", (), {"lookahead_samples": object()})()
        out = offline.run_stream(np.arange(10, dtype=np.float32), runner, block=4, pad=0)
        self.assertEqual(out.shape, (10,))

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


if __name__ == "__main__":
    unittest.main()
