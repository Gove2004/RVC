"""推理运行器的缓冲区契约 — 几何换算、滚动缓冲、formant 档位覆盖。

这些算术此前埋在 init_processing / prewarm_formant 里，只能开流实跑验证。
抽到 rvc.streaming.buffers 后无需 GPU 即可锁住；滚动缓冲用假重采样器在 CPU
张量上跑真实的 _stage_preprocess，锁住写区几何（修正帧 + 新块）与左移拷贝；
真实重采样下的拼接正确性（历史内部与整段重采样一致、逐块无漂移）由
TestResamplerSplicing 用 torchaudio Resample 锁。
"""
import math
import unittest

import numpy as np
import torch
from torchaudio.transforms import Resample

from rvc.core.config import FORMANT_MAX, FORMANT_MIN
from rvc.core.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
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
    # 真实重采样输出 = 新块 + 2 厘秒上下文（B16 + 2*HUBERT_FRAME_SIZE），
    # _stage_preprocess 切掉前 HUBERT_FRAME_SIZE 个左缘样本后剩 B16 + HUBERT_FRAME_SIZE
    # （修正帧 + 新块）。假重采样器按这个原始长度返回零。
    state.resampler_work_to_16k = _FakeResampler(g.block_samples_16k + 2 * HUBERT_FRAME_SIZE)
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

    def test_16k_write_covers_correction_frame_and_new_block(self):
        """写区 = 修正帧（HUBERT_FRAME_SIZE）+ 新块（B16）；写区左侧纯滚动不改写。

        修正帧把上一块尾部（当时缺右支撑的因果前沿）用本块的完整左右支撑重算，
        使其收敛到整段重采样值；每个样本只被修正一次，之后只随缓冲左移拷贝。
        真实重采样下的「内部与整段一致 / 逐块无漂移」由 TestResamplerSplicing 锁。
        """
        runner, g = _rolling_runner()
        state = runner.state
        total = g.buffer_16k_samples
        state.input_wav_16k[:] = torch.arange(total, dtype=torch.float32) + 1.0
        before = state.input_wav_16k.clone()
        runner._stage_preprocess(torch.zeros(g.block_samples))
        write_len = g.block_samples_16k + HUBERT_FRAME_SIZE
        # 写区左侧：纯滚动，等于旧缓冲左移 B16，历史不被改写
        self.assertTrue(torch.equal(
            state.input_wav_16k[:total - write_len],
            before[g.block_samples_16k:total - write_len + g.block_samples_16k]))
        # 写区（修正帧 + 新块）：被重采样输出覆盖（fake 返回零）
        self.assertTrue(torch.equal(
            state.input_wav_16k[-write_len:], torch.zeros(write_len)))

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


def _splice_snapshots(sr, block_time=0.05):
    """用真实 torchaudio Resample 跑 _stage_preprocess，逐块快照 16k 缓冲。

    返回 (snapshots, geometry, full_truth, n_blocks)。full_truth 是整段输入一次性
    重采样的结果（stateless FIR，仅自身左右两端有零垫伪影），作为拼接正确性的基准。
    """
    runner, g = _rolling_runner(sr=sr, block_time=block_time)
    runner.state.resampler_work_to_16k = Resample(sr, HUBERT_SAMPLE_RATE, dtype=torch.float32)
    st = runner.state
    n_blocks = math.ceil(g.buffer_16k_samples / g.block_samples_16k) + 6
    torch.manual_seed(0)
    sig = torch.randn(n_blocks * g.block_samples)
    full = st.resampler_work_to_16k(sig)
    snaps = []
    for k in range(n_blocks):
        runner._stage_preprocess(sig[k * g.block_samples:(k + 1) * g.block_samples])
        snaps.append(st.input_wav_16k.clone())
    return snaps, g, full, n_blocks


class TestResamplerSplicing(unittest.TestCase):
    """真实重采样下的拼接契约：修正帧让历史收敛到整段重采样，且逐块无漂移。

    因果前沿（末块右缘缺右支撑的少数样本）下一块即落入修正帧被修正；每个样本
    只被修正一次，之后只随缓冲左移拷贝。MARGIN 取远大于前沿宽度的余量。
    """

    MARGIN = 32

    def test_history_interior_matches_full_resample(self):
        for sr in (40000, 44100, 48000, 96000):
            with self.subTest(sr=sr):
                snaps, g, full, n = _splice_snapshots(sr)
                b16, length = g.block_samples_16k, g.buffer_16k_samples
                left_abs = n * b16 - length
                got = snaps[-1][:length - self.MARGIN]
                want = full[left_abs:left_abs + length - self.MARGIN]
                self.assertTrue(
                    torch.allclose(got, want, atol=2e-6, rtol=0.0),
                    f"max err {(got - want).abs().max().item():.3e}")

    def test_no_drift_across_blocks(self):
        """相邻块快照在「重叠且越过因果前沿」的绝对样本上必须比特级一致。

        越过前沿的样本要么已被修正、要么是修正值的拷贝，都等于同一个卷积输出，
        故 torch.equal 成立——这锁死「修正一次后不再变化」，杜绝逐块累积漂移。
        """
        for sr in (40000, 44100, 48000, 96000):
            with self.subTest(sr=sr):
                snaps, g, _, n = _splice_snapshots(sr)
                b16, length = g.block_samples_16k, g.buffer_16k_samples
                for k in range(n - 1):
                    ov_left = (k + 2) * b16 - length
                    end = (k + 1) * b16 - self.MARGIN
                    if ov_left < 0 or end <= ov_left:
                        continue
                    a0 = ov_left - ((k + 1) * b16 - length)
                    b0 = ov_left - ((k + 2) * b16 - length)
                    seg_a = snaps[k][a0:a0 + (end - ov_left)]
                    seg_b = snaps[k + 1][b0:b0 + (end - ov_left)]
                    self.assertTrue(
                        torch.equal(seg_a, seg_b),
                        f"block {k}->{k + 1} drift {(seg_a - seg_b).abs().max().item():.3e}")


if __name__ == "__main__":
    unittest.main()