"""固定噪声场必须随窗口滚动，保持与绝对时间对齐。

## 为什么需要这个测试

`state.noise_z` 是 z_p 采样用的固定噪声场，形状 = m_p 的形状（100fps）。它按
**缓冲区索引**存放，而输入窗口每块前移 block_samples。若不随窗口一起左移，
同一段绝对音频在相邻块会落到不同索引、取到不同噪声。

实测（真实 base 变体、0.25s 块、凤凰.wav）：不滚动时同一绝对时刻的 z_p 相对差
112%，而"完全换一份独立随机噪声"的上界是 127% —— 固定噪声场在滑动窗口下几乎
没起作用，等价于逐块 randn_like。这与 state.py 里"内容不变则渲染不变"的承诺相反。

同时注意：噪声项比 m_p 的信号项大（实测范数比 2.56），所以它主导 z_p 的块间波动，
不是可以忽略的小量。

## 本文件钉住什么

1. 滚动语义：保留段逐位来自原位，尾部换新
2. 分布不变：多次滚动后仍 ≈ N(0,1)（否则等同于改了音色统计）
3. advance 与窗口前移量一致
4. 边界：advance ≥ 长度时整场重抽，不越界
5. noise_z 为 None 时不报错
6. 接线：_stage_preprocess 必须调用 _roll_noise_field（防止被静默摘掉）
"""
import ast
import inspect
import unittest

import torch

from rvc.pipeline.state import EngineState
from rvc.streaming.runner import InferenceRunner


def _make_runner(block_samples=10000, samples_per_centis=400, channels=4, length=56):
    """构造不跑 initialize 的最小 runner，只用于测滚动函数。"""
    st = EngineState()
    st.device = torch.device('cpu')
    st.block_samples = block_samples
    st.samples_per_centis = samples_per_centis
    st.noise_z = torch.randn(1, channels, length)
    st.noise_scratch = torch.empty_like(st.noise_z)
    obj = InferenceRunner.__new__(InferenceRunner)
    obj.state = st
    return obj


class TestRollSemantics(unittest.TestCase):
    def test_kept_region_is_bitwise_identical(self):
        r = _make_runner()
        adv = r.state.block_samples // r.state.samples_per_centis
        before = r.state.noise_z.clone()
        length = r.state.noise_z.shape[-1]

        r._roll_noise_field()

        self.assertEqual(adv, 25)
        self.assertTrue(
            torch.equal(r.state.noise_z[..., :length - adv], before[..., adv:]),
            '保留段必须逐位等于滚动前的对应位置，否则同一绝对音频会换噪声',
        )

    def test_tail_is_refilled(self):
        r = _make_runner()
        adv = r.state.block_samples // r.state.samples_per_centis
        before = r.state.noise_z.clone()
        length = r.state.noise_z.shape[-1]

        r._roll_noise_field()

        self.assertFalse(
            torch.equal(r.state.noise_z[..., length - adv:], before[..., length - adv:]),
            '尾部对应新进入窗口的音频，必须换新随机值',
        )

    def test_double_roll_equals_double_shift(self):
        """连滚两次 == 一次移 2*adv（保留段仍逐位一致）。"""
        r = _make_runner()
        adv = r.state.block_samples // r.state.samples_per_centis
        length = r.state.noise_z.shape[-1]
        before = r.state.noise_z.clone()

        r._roll_noise_field()
        r._roll_noise_field()

        self.assertTrue(torch.equal(
            r.state.noise_z[..., :length - 2 * adv], before[..., 2 * adv:]))


class TestDistributionPreserved(unittest.TestCase):
    def test_many_rolls_stay_standard_normal(self):
        """音色不能被改动：噪声仍是标准正态。"""
        r = _make_runner(channels=64, length=256)
        for _ in range(200):
            r._roll_noise_field()
        nz = r.state.noise_z
        self.assertLess(abs(nz.mean().item()), 0.02, f'均值漂移: {nz.mean().item()}')
        self.assertLess(abs(nz.std().item() - 1.0), 0.02, f'标准差漂移: {nz.std().item()}')

    def test_shape_unchanged(self):
        r = _make_runner()
        shape = tuple(r.state.noise_z.shape)
        r._roll_noise_field()
        self.assertEqual(tuple(r.state.noise_z.shape), shape)


class TestEdges(unittest.TestCase):
    def test_advance_ge_length_regenerates_without_error(self):
        """块长 ≥ 噪声场长度时整场重抽，不越界、不报错。"""
        r = _make_runner(block_samples=40000, samples_per_centis=400, length=56)
        # advance = 100 ≥ length 56
        before = r.state.noise_z.clone()
        r._roll_noise_field()
        self.assertEqual(tuple(r.state.noise_z.shape), tuple(before.shape))
        self.assertFalse(torch.equal(r.state.noise_z, before))
        self.assertLess(abs(r.state.noise_z.std().item() - 1.0), 0.5)

    def test_none_noise_is_noop(self):
        r = _make_runner()
        r.state.noise_z = None
        r.state.noise_scratch = None
        r._roll_noise_field()  # 不应抛异常
        self.assertIsNone(r.state.noise_z)

    def test_zero_advance_is_noop(self):
        """block_samples < samples_per_centis（不足一帧）时不动。"""
        r = _make_runner(block_samples=100, samples_per_centis=400)
        before = r.state.noise_z.clone()
        r._roll_noise_field()
        self.assertTrue(torch.equal(r.state.noise_z, before))


class TestWiring(unittest.TestCase):
    def test_preprocess_calls_roll(self):
        """_stage_preprocess 必须滚动噪声场，否则固定噪声场会静默失效。"""
        src = inspect.getsource(InferenceRunner._stage_preprocess)
        self.assertIn('_roll_noise_field', src,
                      '_stage_preprocess 未调用 _roll_noise_field：'
                      '噪声场将不再随窗口滚动，同一音频每块拿到不同噪声')

    def test_roll_called_before_buffer_roll(self):
        """顺序：先滚噪声再滚缓冲区（两者必须同步，顺序本身不影响结果，
        但显式钉住可以避免未来插入的代码把二者拆开）。"""
        fn = InferenceRunner._stage_preprocess
        tree = ast.parse(inspect.getsource(fn).lstrip())
        body_src = inspect.getsource(fn)
        i_noise = body_src.index('_roll_noise_field')
        i_buf = body_src.index('input_wav_16k_swap')
        self.assertLess(i_noise, i_buf)
        self.assertIsNotNone(tree)


if __name__ == '__main__':
    unittest.main()
