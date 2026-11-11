"""推理运行器的缓冲区契约 — 几何换算、滚动缓冲、formant 档位覆盖。

这些算术此前埋在 init_processing / prewarm_formant 里，只能开流实跑验证。
抽到 rvc.streaming.buffers 后无需 GPU 即可锁住；滚动缓冲用假重采样器在 CPU
张量上跑真实的 _stage_preprocess，锁住"历史不被改写"这条最易回归的性质。
"""
import unittest

import numpy as np
import torch

from rvc.core.config import FORMANT_MAX, FORMANT_MIN
from rvc.core.constants import HUBERT_FRAME_SIZE
from rvc.pipeline.state import EngineState
from rvc.pipeline.synthesis import formant_frames
from rvc.streaming.buffers import buffer_geometry, formant_frame_range
from rvc.streaming.runner import InferenceRunner


class TestBufferGeometry(unittest.TestCase):
    def test_lengths_align_to_10ms(self):
        for sr in (40000, 44100, 48000, 96000):
            with self.subTest(sr=sr):
                g = buffer_geometry(sr, 0.2537, 0.1173, 2.3456)
                zc = sr // 100
                for name in ("block_samples", "crossfade_samples", "extra_samples",
                             "sola_buffer_samples", "sola_search_samples"):
                    self.assertEqual(getattr(g, name) % zc, 0, name)

    def test_sola_overlap_capped_at_100ms(self):
        """上限只约束"能调多长"，用户自己的 crossfade_time 更短时原样保留。"""
        g = buffer_geometry(40000, 0.25, 0.50, 2.0)
        self.assertEqual(g.sola_buffer_samples, 10 * 400)
        self.assertEqual(g.crossfade_samples, 20000)
        short = buffer_geometry(40000, 0.25, 0.04, 2.0)
        self.assertEqual(short.sola_buffer_samples, short.crossfade_samples)

    def test_return_length_is_block_plus_sola(self):
        g = buffer_geometry(48000, 0.25, 0.10, 2.0)
        self.assertEqual(
            g.return_length * g.samples_per_centis,
            g.block_samples + g.sola_buffer_samples + g.sola_search_samples,
        )
        self.assertEqual(g.skip_head, g.extra_samples // g.samples_per_centis)

    def test_16k_lengths_scale_by_frame_size(self):
        for sr in (40000, 44100, 48000, 96000):
            with self.subTest(sr=sr):
                g = buffer_geometry(sr, 0.25, 0.10, 2.0)
                self.assertEqual(g.block_samples_16k,
                                 HUBERT_FRAME_SIZE * g.block_samples // g.samples_per_centis)
                self.assertEqual(g.buffer_16k_samples,
                                 HUBERT_FRAME_SIZE * g.work_buffer_samples // g.samples_per_centis)

    def test_work_buffer_holds_extra_crossfade_search_block(self):
        g = buffer_geometry(40000, 0.25, 0.10, 2.0)
        self.assertEqual(g.work_buffer_samples,
                         g.extra_samples + g.crossfade_samples
                         + g.sola_search_samples + g.block_samples)


class TestFormantFrameRange(unittest.TestCase):
    def test_covers_every_quantized_frame_in_range(self):
        """预捕获少覆盖一档，运行中拖到那档就会在回调线程里现场捕获（断音）。"""
        for rl in (1, 7, 30, 36, 101):
            with self.subTest(return_length=rl):
                covered = set(formant_frame_range(rl, FORMANT_MIN, FORMANT_MAX))
                for semitone in np.linspace(FORMANT_MIN, FORMANT_MAX, 201):
                    factor = 2.0 ** (float(semitone) / 12.0)
                    frames, _ = formant_frames(factor, rl)
                    self.assertIn(frames, covered)

    def test_matches_formant_frames_at_both_ends(self):
        rl = 36
        rng = formant_frame_range(rl, FORMANT_MIN, FORMANT_MAX)
        self.assertEqual(rng.start, formant_frames(2.0 ** (FORMANT_MIN / 12.0), rl)[0])
        self.assertEqual(rng.stop - 1, formant_frames(2.0 ** (FORMANT_MAX / 12.0), rl)[0])

    def test_never_below_one_frame(self):
        self.assertGreaterEqual(formant_frame_range(1, FORMANT_MIN, FORMANT_MAX).start, 1)


class _FakeResampler:
    """记录输入长度、返回定长零，替代 work_sr→16k 的 TatResample。"""

    def __init__(self, out_len):
        self.out_len = out_len
        self.input_lengths = []

    def __call__(self, x):
        self.input_lengths.append(int(x.numel()))
        return torch.zeros(self.out_len, dtype=torch.float32)


def _rolling_runner(sr=40000, block_time=0.10, crossfade_time=0.10, extra_time=0.50):
    """在 CPU 上搭一个只够跑 _stage_preprocess 的 runner。"""
    g = buffer_geometry(sr, block_time, crossfade_time, extra_time)
    state = EngineState(device="cpu")
    state.work_sr = sr
    state.samples_per_centis = g.samples_per_centis
    state.block_samples = g.block_samples
    state.block_samples_16k = g.block_samples_16k
    state.input_wav_work = torch.zeros(g.work_buffer_samples)
    state.input_wav_work_swap = torch.empty_like(state.input_wav_work)
    state.input_wav_16k = torch.zeros(g.buffer_16k_samples)
    state.input_wav_16k_swap = torch.empty_like(state.input_wav_16k)
    # 输出比真实重采样多 HUBERT_FRAME_SIZE 个样本，供 _stage_preprocess 切掉头部
    state.resampler_work_to_16k = _FakeResampler(g.block_samples_16k + HUBERT_FRAME_SIZE)
    runner = InferenceRunner.__new__(InferenceRunner)
    runner.state = state
    return runner, g


class TestRollingBuffers(unittest.TestCase):
    def test_work_buffer_keeps_block_history_in_order(self):
        runner, g = _rolling_runner()
        state = runner.state
        blocks = [torch.full((g.block_samples,), float(i + 1)) for i in range(12)]
        for b in blocks:
            runner._stage_preprocess(b)
        # 缓冲长度不必是整块数，只校验末尾若干个整块的顺序与内容
        n_whole = g.work_buffer_samples // g.block_samples
        expected = torch.cat(blocks[-n_whole:])
        self.assertTrue(torch.equal(state.input_wav_work[-expected.numel():], expected))

    def test_16k_write_only_touches_current_block(self):
        """重采样输出只写当前块尾部；写进修正帧会改写上一块尾部，让 HuBERT
        看到的历史每块都被"修正"，连续累积后特征漂移。"""
        runner, g = _rolling_runner()
        state = runner.state
        total = g.buffer_16k_samples
        state.input_wav_16k[:] = torch.arange(total, dtype=torch.float32) + 1.0
        before = state.input_wav_16k.clone()
        runner._stage_preprocess(torch.zeros(g.block_samples))
        shifted = before[g.block_samples_16k:]
        self.assertTrue(torch.equal(
            state.input_wav_16k[:total - g.block_samples_16k], shifted))
        self.assertTrue(torch.equal(
            state.input_wav_16k[-g.block_samples_16k:],
            torch.zeros(g.block_samples_16k)))

    def test_resampler_sees_two_centis_of_context(self):
        runner, g = _rolling_runner()
        runner._stage_preprocess(torch.zeros(g.block_samples))
        self.assertEqual(runner.state.resampler_work_to_16k.input_lengths,
                         [g.block_samples + 2 * g.samples_per_centis])

    def test_double_buffer_swaps_objects(self):
        runner, _ = _rolling_runner()
        state = runner.state
        work_a, work_b = state.input_wav_work, state.input_wav_work_swap
        k_a, k_b = state.input_wav_16k, state.input_wav_16k_swap
        runner._stage_preprocess(torch.zeros(state.block_samples))
        self.assertIs(state.input_wav_work, work_b)
        self.assertIs(state.input_wav_work_swap, work_a)
        self.assertIs(state.input_wav_16k, k_b)
        self.assertIs(state.input_wav_16k_swap, k_a)


if __name__ == "__main__":
    unittest.main()