"""rvc.dsp.realism — 真实感搬运链的单元测试（CPU）。

锁定的契约：
- 全 0 强度 = 逐位旁路（不碰信号，只钳位）
- 每一级单独打开时，改动方向正确且只在目标频带
- distance 的电平起伏有界（不会「越说越小声」）、可复现（同 seed 同序列）
- reset() 恢复到初始状态（滤波器历史、游走、RNG 全清）
- NaN/Inf 输入不产生 NaN 输出；输出恒在 [-1, 1]
- 稳态零增长：块长不变时工作缓冲不重建
"""
import unittest

import torch

from rvc.dsp.realism import RealismChain, RealismParams

SR = 16000
BLOCK = 4096


def _tone(freq: float, amp: float = 0.5, n: int = BLOCK) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / SR
    return amp * torch.sin(2 * torch.pi * freq * t)


def _noise(amp: float = 0.1, n: int = BLOCK, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return amp * torch.randn(n, generator=g)


def _band_energy(x: torch.Tensor, lo: float, hi: float) -> float:
    """[lo, hi) 频带的平均功率（rfft 幅值平方的均值）。"""
    spec = torch.fft.rfft(x).abs() ** 2
    freqs = torch.fft.rfftfreq(x.shape[0], 1.0 / SR)
    return float(spec[(freqs >= lo) & (freqs < hi)].mean())


def _chain(**kw) -> RealismChain:
    c = RealismChain(RealismParams(**kw))
    c.setup(SR, BLOCK, "cpu")
    return c


class TestBypass(unittest.TestCase):
    def test_default_params_are_noop(self):
        """默认全 0：输出必须与输入逐位相同（不是「近似相同」）。"""
        out = _tone(220, amp=0.4)
        res = _chain().process(out.clone(), _noise())
        self.assertTrue(torch.equal(res, out))

    def test_zero_params_noop_after_active(self):
        """跑过带强度的块后再把强度清零，应立刻回到旁路（状态不残留到输出）。"""
        c = _chain(breath=1.0, residual=1.0, rumble=1.0, distance=1.0)
        out = _tone(220, amp=0.4)
        c.process(out.clone(), _noise())
        c.params = RealismParams()
        self.assertTrue(torch.equal(c.process(out.clone(), _noise()), out))

    def test_process_before_setup_is_safe(self):
        """未 setup 时不处理但保证有限（离线路径可能先于 setup 调用）。"""
        out = _tone(220, amp=0.4)
        res = RealismChain().process(out.clone(), _noise())
        self.assertTrue(torch.equal(res, out))

    def test_reset_before_setup_is_safe(self):
        RealismChain().reset()  # 不应抛异常


class TestBreath(unittest.TestCase):
    def test_raises_high_band(self):
        """宽带参考 → >4kHz 能量显著上升。"""
        out = _tone(220, amp=0.4)
        before = _band_energy(out, 5000, 7500)
        res = _chain(breath=1.0).process(out.clone(), _noise(amp=0.1))
        self.assertGreater(_band_energy(res, 5000, 7500), before * 10)

    def test_gated_by_reference_high_band(self):
        """参考本身没有高频（纯元音样）→ 几乎不注入，避免把底噪抹到齿音上。"""
        out = _tone(220, amp=0.4)
        clean = _band_energy(out, 5000, 7500)
        res = _chain(breath=1.0).process(out.clone(), _tone(220, amp=0.3))
        self.assertLess(_band_energy(res, 5000, 7500), clean * 1.2)


class TestResidual(unittest.TestCase):
    def test_raises_noise_floor(self):
        """噪声参考 → 输出的噪声底（宽带能量）显著上升。"""
        out = _tone(220, amp=0.4)
        before = _band_energy(out, 5000, 7500)
        res = _chain(residual=1.0).process(out.clone(), _noise(amp=0.1))
        self.assertGreater(_band_energy(res, 5000, 7500), before * 10)

    def test_tonal_reference_injects_little(self):
        """参考是干净谐波时底噪回注量很小：干净进、干净出。"""
        out = _tone(220, amp=0.4)
        clean = _band_energy(out, 5000, 7500)
        noisy = _band_energy(_chain(residual=1.0).process(out.clone(), _noise(amp=0.1)), 5000, 7500)
        tonal = _band_energy(_chain(residual=1.0).process(out.clone(), _tone(220, amp=0.3)), 5000, 7500)
        self.assertLess(tonal, clean * 2)
        self.assertGreater(noisy, tonal * 20, "噪声样程度应主导注入量")

    def test_fills_silent_gap_with_floor(self):
        """输出静音、输入只剩底噪时仍注入底噪。

        底噪最该出现的位置就是间隙：若换算系数直接取当前块 RMS，间隙里
        系数为 0，静音段依旧「死静」，这一级就白做了。
        """
        c = _chain(residual=1.0)
        speech = _tone(220, amp=0.4)
        for _ in range(3):
            c.process(speech.clone(), speech + _noise(amp=0.01))
        gap = c.process(torch.zeros(BLOCK), _noise(amp=0.01))
        gap_rms = float(gap.pow(2).mean().sqrt())
        self.assertTrue(torch.isfinite(gap).all())
        self.assertGreater(gap_rms, 0.002, "间隙里没有底噪，等于这一级没生效")
        self.assertLess(gap_rms, 0.05, "间隙里底噪过响")


class TestRumble(unittest.TestCase):
    def test_raises_low_band(self):
        out = _tone(80, amp=0.1)
        before = _band_energy(out, 20, 200)
        res = _chain(rumble=1.0).process(out.clone(), _tone(80, amp=0.3))
        self.assertGreater(_band_energy(res, 20, 200), before * 1.5)

    def test_gated_by_input_proximity(self):
        """近讲度取自输入的低频/中频比：低频重的参考提升多，明亮的参考几乎不动。"""
        out = _tone(80, amp=0.1)
        before = _band_energy(out, 20, 200)
        near = _band_energy(_chain(rumble=1.0).process(out.clone(), _tone(80, amp=0.3)), 20, 200)
        far = _band_energy(_chain(rumble=1.0).process(out.clone(), _tone(2000, amp=0.3)), 20, 200)
        self.assertGreater(near, before * 1.5)
        self.assertLess(far, near * 0.5)

    def test_stays_finite_on_loud_block(self):
        """软饱和兜住峰值：满幅低频块不应越界或产生非有限值。"""
        loud = _tone(80, amp=0.99)
        res = _chain(rumble=1.0).process(loud.clone(), _tone(80, amp=0.99))
        self.assertTrue(torch.isfinite(res).all())
        self.assertLessEqual(float(res.abs().max()), 1.0)


class TestDistance(unittest.TestCase):
    def _gains(self, chain: RealismChain, n_blocks: int, sig: torch.Tensor) -> list[float]:
        ref_rms = float(sig.pow(2).mean().sqrt())
        return [float(chain.process(sig.clone(), sig).pow(2).mean().sqrt()) / ref_rms
                for _ in range(n_blocks)]

    def test_gain_is_bounded_and_modulated(self):
        """电平起伏在 ±1.5dB（含倾斜余量）内，且确实在动（不是恒定）。"""
        gains = self._gains(_chain(distance=1.0), 24, _tone(100, amp=0.2))
        self.assertTrue(all(0.78 <= g <= 1.28 for g in gains), f"越界: {gains}")
        self.assertGreater(max(gains) - min(gains), 0.01, "电平没有起伏，distance 等于没开")

    def test_long_run_does_not_drift(self):
        """长跑不漂移：游走有界，电平不会越说越小。"""
        gains = self._gains(_chain(distance=1.0), 120, _tone(100, amp=0.2))
        self.assertTrue(all(0.78 <= g <= 1.28 for g in gains), "游走漂出量程")
        self.assertGreater(sum(gains[-50:]) / 50, 0.85, "长跑后电平持续偏低 = 游走贴边")

    def test_zero_distance_is_noop(self):
        sig = _tone(100, amp=0.2)
        self.assertTrue(torch.equal(_chain(distance=0.0).process(sig.clone(), sig), sig))

    def test_same_seed_is_deterministic(self):
        sig = _tone(100, amp=0.2)
        a = self._gains(_chain(distance=1.0), 8, sig)
        b = self._gains(_chain(distance=1.0), 8, sig)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        sig = _tone(100, amp=0.2)
        a = _chain(distance=1.0)
        b = RealismChain(RealismParams(distance=1.0), seed=12345)
        b.setup(SR, BLOCK, "cpu")
        self.assertNotEqual(self._gains(a, 8, sig), self._gains(b, 8, sig))


class TestState(unittest.TestCase):
    def test_reset_restores_initial_state(self):
        """reset 后重放同一串块，输出与首次完全一致（历史/游走/RNG 全清）。"""
        c = _chain(breath=1.0, residual=1.0, rumble=1.0, distance=1.0)
        out, ref = _tone(220, amp=0.4), _noise(amp=0.1)
        first = [c.process(out.clone(), ref).clone() for _ in range(4)]
        c.reset()
        again = [c.process(out.clone(), ref) for _ in range(4)]
        for i, (a, b) in enumerate(zip(first, again)):
            self.assertTrue(torch.equal(a, b), f"第 {i} 块在 reset 后不一致")

    def test_reset_clears_filter_history(self):
        """reset 清空 FIR 尾部：首块与全新链的首块相同。"""
        out, ref = _tone(220, amp=0.4), _noise(amp=0.1)
        c = _chain(breath=1.0)
        for _ in range(3):
            c.process(out.clone(), ref)
        c.reset()
        fresh = _chain(breath=1.0)
        self.assertTrue(torch.equal(c.process(out.clone(), ref), fresh.process(out.clone(), ref)))

    def test_buffers_are_reused_not_grown(self):
        """稳态零增长：块长不变时工作缓冲对象不重建。"""
        c = _chain(breath=1.0, residual=1.0, rumble=1.0, distance=1.0)
        out, ref = _tone(220, amp=0.4), _noise(amp=0.1)
        c.process(out.clone(), ref)
        work, n = c._ref_work, c._n
        for _ in range(20):
            c.process(out.clone(), ref)
        self.assertIs(c._ref_work, work)
        self.assertEqual(c._n, n)

    def test_block_length_change_is_handled(self):
        c = _chain(breath=1.0)
        res = c.process(torch.zeros(1024), torch.zeros(1024))
        self.assertEqual(res.shape[0], 1024)


class TestSafety(unittest.TestCase):
    def test_nan_input_produces_finite_output(self):
        """NaN/Inf 参考与输出都被消毒：本块输出有限，且不跨块传播。"""
        c = _chain(breath=1.0, residual=1.0, rumble=1.0, distance=1.0)
        bad_out = _tone(220, amp=0.4)
        bad_out[10] = float("nan")
        bad_out[20] = float("inf")
        bad_ref = _noise(amp=0.1)
        bad_ref[30] = float("nan")
        res = c.process(bad_out, bad_ref)
        self.assertTrue(torch.isfinite(res).all(), "NaN 泄漏到输出")
        nxt = c.process(_tone(220, amp=0.4), _noise(amp=0.1))
        self.assertTrue(torch.isfinite(nxt).all(), "NaN 经内部状态跨块传播")

    def test_output_is_clamped(self):
        """越界输入（音量增益 > 1 之后）也被钳在 [-1, 1]。"""
        out = _tone(220, amp=2.5)
        ref = _noise(amp=1.0)
        res = _chain(breath=1.0, residual=1.0, rumble=1.0, distance=1.0).process(out, ref)
        self.assertLessEqual(float(res.abs().max()), 1.0)
        self.assertTrue(torch.isfinite(res).all())

    def test_shape_and_dtype_preserved(self):
        out = _tone(220, amp=0.4)
        res = _chain(breath=1.0).process(out, _noise())
        self.assertEqual(res.shape, out.shape)
        self.assertEqual(res.dtype, torch.float32)

    def test_shorter_reference_than_output(self):
        """参考块短于输出块时补零而不是抛异常（块长契约由调用方保证，此处兜底）。"""
        res = _chain(residual=1.0).process(_tone(220, amp=0.4), _noise(n=100))
        self.assertEqual(res.shape[0], BLOCK)
        self.assertTrue(torch.isfinite(res).all())


if __name__ == "__main__":
    unittest.main()
