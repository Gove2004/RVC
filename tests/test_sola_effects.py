"""SOLA 拼接与输出效果链 — CPU 单元测试。

锁定的核心契约：
- apply_sola 输出恒为 block_samples 定长（SOLA 对齐契约）
- 平滑周期信号的连续块经 SOLA 拼接后无跳变（不产生咔哒/重影）
- NaN 防护：buffer 污染后输出仍有限值，buffer 复位为零
- RmsMixEffect：rms_mix=1 恒等旁路；rms_mix=0 跟随参考响度
- MicAirflowEffect：强度 0 恒等旁路；输出不随分块长度漂移（ulp 级）；强度单调
- AudioProcessor 全链可在 CPU 上 setup/process 往返
"""
import unittest

import numpy as np
import torch

from rvc.dsp.sola import apply_sola
from rvc.streaming.effects import AudioProcessor, MicAirflowEffect, RmsMixEffect, SolaEffect

BLOCK = 1000
BUF = 100
SEARCH = 100


def _make_sola_effect() -> SolaEffect:
    eff = SolaEffect()
    eff.setup(block_samples=BLOCK, sola_buffer_samples=BUF,
              sola_search_samples=SEARCH, device="cpu")
    return eff


def _periodic_blocks(n_blocks: int):
    """生成连续周期信号的推理块序列（模拟合成器输出：块长 = block+buf+search）。

    周期 200 样本整除搜索窗，保证每块最优 offset 都落在整数周期上，
    SOLA 拼接后应还原出连续正弦（无跳变）。
    """
    period = 200
    total = n_blocks * BLOCK + BUF + SEARCH
    t = np.arange(total, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * t / period)
    blocks = []
    for i in range(n_blocks):
        start = i * BLOCK
        blocks.append(torch.from_numpy(x[start:start + BLOCK + BUF + SEARCH].copy()))
    return x, blocks


class TestApplySola(unittest.TestCase):
    def test_zero_overlap_returns_block_without_crash(self):
        """crossfade=0（空核）不能触发 F.conv1d kernel width=0 崩溃。"""
        eff = SolaEffect()
        eff.setup(block_samples=BLOCK, sola_buffer_samples=0,
                  sola_search_samples=SEARCH, device="cpu")
        x = torch.randn(BLOCK + SEARCH)
        out = eff.process(x)
        self.assertEqual(out.shape[0], BLOCK)
        self.assertTrue(torch.equal(out, x[:BLOCK]))

    def test_output_is_fixed_length(self):
        eff = _make_sola_effect()
        _, blocks = _periodic_blocks(3)
        for b in blocks:
            out = eff.process(b)
            self.assertEqual(out.shape[0], BLOCK)

    def test_concatenated_output_is_smooth(self):
        """连续周期块拼接后无跳变：相邻样本差不超过正弦自身斜率的容差。"""
        eff = _make_sola_effect()
        n = 6
        _, blocks = _periodic_blocks(n)
        outs = [eff.process(b) for b in blocks]
        joined = torch.cat(outs).numpy()
        diff = np.abs(np.diff(joined))
        # 正弦 amp=0.5、period=200 的最大斜率 ≈ 0.0157；拼接若产生跳变
        # （重复/缺段）会在此处留下接近幅度级的差分。给足余量。
        self.assertLess(diff.max(), 0.05,
                        f"SOLA 拼接产生跳变，max|diff|={diff.max():.4f}")

    def test_block_smaller_than_buffer_reconstructs_exactly(self):
        """block<buffer 时 head=infer[:buf] 与 tail=infer[block:block+buf] 视图重叠。

        原地交叉淡化 infer[:buf]*=fade_in 会改写 [block:buf] 段（属 tail），若先淡化
        再取 tail 存入 buffer，下一块参考被污染，稳态输出偏离真值（实测 block100/
        buf400 偏差 0.34，信号幅 0.5）。连续周期信号 offset=0 的理想还原应逐样本
        等于原信号——锁住「先取干净 tail 再淡化」。
        """
        block, buf, search = 100, 400, 100
        eff = SolaEffect()
        eff.setup(block_samples=block, sola_buffer_samples=buf,
                  sola_search_samples=search, device="cpu")
        n, period = 12, 200
        total = n * block + buf + search
        t = np.arange(total, dtype=np.float32)
        x = 0.5 * np.sin(2 * np.pi * t / period)
        outs = []
        for k in range(n):
            s = k * block
            infer = torch.from_numpy(x[s:s + block + buf + search].copy())
            outs.append(eff.process(infer))
        out = torch.cat(outs).numpy()
        ideal = x[:n * block]
        # 跳过启动块（buffer 零淡入），稳态须逐样本还原
        np.testing.assert_allclose(out[block:], ideal[block:], atol=1e-5)

    def test_nan_in_buffer_is_contained(self):
        """buffer 被 NaN 污染：本块输出仍有限、buffer 被新块有限 tail 覆盖。

        契约：NaN buffer 在交叉淡化混合时被零替代（本块不受污染）；
        buffer 更新取新块 tail（有限），NaN 不会经 buffer 跨块传播。
        """
        eff = _make_sola_effect()
        _, blocks = _periodic_blocks(2)
        eff.process(blocks[0])
        eff.sola_buffer.fill_(float("nan"))
        out = eff.process(blocks[1])
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(eff.sola_buffer).all())

    def test_nan_in_infer_head_does_not_persist(self):
        """infer 头部含 NaN：本块输出不可救（信号本身坏），但 tail 有限使
        buffer 保持有限，下一干净块输出恢复正常——NaN 不跨块持续。"""
        eff = _make_sola_effect()
        _, blocks = _periodic_blocks(3)
        bad = blocks[1].clone()
        bad[:BUF] = float("nan")
        eff.process(blocks[0])
        eff.process(bad)
        self.assertTrue(torch.isfinite(eff.sola_buffer).all())
        out_next = eff.process(blocks[2])
        self.assertTrue(torch.isfinite(out_next).all(),
                        "NaN 经 buffer 跨块传播——违反防护设计目标")

    def test_reset_clears_buffer(self):
        eff = _make_sola_effect()
        _, blocks = _periodic_blocks(1)
        eff.process(blocks[0])
        eff.reset()
        self.assertTrue((eff.sola_buffer == 0).all())


class TestRmsMixEffect(unittest.TestCase):
    def test_mix_one_is_identity_bypass(self):
        eff = RmsMixEffect()
        eff.setup(16000)
        infer = torch.linspace(-1, 1, 1000)
        ref = torch.zeros(1000)
        out = eff.process(infer, ref, 1.0)
        self.assertIs(out, infer, "rms_mix=1 应恒等旁路（不分配）")

    def test_mix_zero_follows_quiet_reference(self):
        eff = RmsMixEffect()
        eff.setup(16000)
        infer = 0.5 * torch.sin(2 * torch.pi * 220 * torch.arange(8000) / 16000)
        ref = 0.02 * torch.sin(2 * torch.pi * 220 * torch.arange(8000) / 16000)
        out = eff.process(infer, ref, 0.0)
        rms_out = float(out.pow(2).mean().sqrt())
        rms_ref = float(ref.pow(2).mean().sqrt())
        self.assertLess(rms_out, rms_ref * 1.5)


class TestAudioProcessorChain(unittest.TestCase):
    def test_setup_process_roundtrip_on_cpu(self):
        """全链（RMS→SOLA）在 CPU 上 setup+连续处理，长度正确、输出有限。"""
        ap = AudioProcessor()
        ap.setup(sr=16000, block_samples=BLOCK, sola_buffer_samples=BUF,
                 sola_search_samples=SEARCH, device="cpu")
        _, blocks = _periodic_blocks(3)
        ref = torch.zeros(BLOCK)
        for b in blocks:
            out = ap.process_output(b, ref, rms_mix=1.0)
            self.assertEqual(out.shape[0], BLOCK)
            self.assertTrue(torch.isfinite(out).all())

    def test_reset_is_safe_before_setup(self):
        ap = AudioProcessor()
        ap.reset()  # 未 setup 时 reset 不应抛异常（离线 process_file 会先 reset）


def _speech_with_pauses(n: int, sr: int) -> torch.Tensor:
    """「说话 0.4s + 近静音 0.4s」交替信号——驱动气流效果的吸气节奏。"""
    seg = int(0.4 * sr)
    t = torch.arange(seg, dtype=torch.float32) / sr
    voiced = 0.3 * torch.sin(2 * torch.pi * 220 * t)
    parts = []
    for _ in range(n // (2 * seg) + 1):
        parts.append(voiced)
        parts.append(torch.full((seg,), 1e-4, dtype=torch.float32))
    return torch.cat(parts)[:n]


class TestMicAirflowEffect(unittest.TestCase):
    """麦克风气流声 — 效果器最硬的契约是**与分块无关**：噪声流、输入上下文缓存、
    仿射递推尾值任一按块长漂移，实时链路就会和离线渲染不一样。
    """

    SR = 16000

    def _effect(self) -> MicAirflowEffect:
        eff = MicAirflowEffect()
        eff.setup(self.SR, "cpu")
        return eff

    def test_zero_intensity_is_identity_bypass(self):
        eff = self._effect()
        x = _speech_with_pauses(40000, self.SR)
        self.assertIs(eff.process(x, 0.0), x, "强度 0 应恒等旁路（不分配、也不推进块间状态）")

    def test_chunking_does_not_change_output(self):
        x = _speech_with_pauses(48000, self.SR)
        whole = self._effect().process(x, 0.7)
        for block in (1600, 3000, 8000):
            eff = self._effect()
            outs = [eff.process(x[i:i + block], 0.7) for i in range(0, x.shape[0], block)]
            drift = float((torch.cat(outs) - whole).abs().max())
            # 不是逐比特：不同输入长度下 conv1d 的浮点结合序会变，差异在 ulp 量级。
            # 真正的契约是「不随块长漂移」——噪声流、上下文缓存、递推尾值任一处
            # 分块敏感的缺陷都会把漂移推到听得出来的量级。
            self.assertLess(drift, 1e-5, f"块长 {block} 与整段渲染漂移 {drift}")

    def test_strength_is_monotonic_and_output_finite(self):
        x = _speech_with_pauses(40000, self.SR)
        deltas = []
        for strength in (0.3, 0.7, 1.0):
            out = self._effect().process(x, strength)
            self.assertTrue(torch.isfinite(out).all())
            deltas.append(float((out - x).pow(2).mean().sqrt()))
        self.assertTrue(deltas[0] < deltas[1] < deltas[2], f"强度应单调递增：{deltas}")

    def test_inhale_arc_fades_before_speech_resumes(self):
        """吸气弧必须在停顿内部走完：开口瞬间的气息能量要明显低于停顿里的峰值。

        反例是 git 版的「升到开口那一刻才到顶」——呼呼压在每个停顿后的第一个字上
        再被 τ=0.05s 硬切，实测断句处比停顿峰值只低 0.6 dB（听感=位置错+突兀）。
        """
        sr, eff = self.SR, self._effect()
        voiced = 0.3 * torch.sin(
            2 * torch.pi * 220 * torch.arange(int(0.6 * sr), dtype=torch.float32) / sr)
        gap = torch.full((int(0.8 * sr),), 1e-4, dtype=torch.float32)
        x = torch.cat([voiced, gap, voiced])
        y = eff.process(x, 1.0)
        ins = (y - x).pow(2)

        def win_rms(a, b):
            return float(ins[int(a * sr):int(b * sr)].mean().sqrt())

        gap_end = 0.6 + 0.8
        peak_gap = max(win_rms(s, s + 0.15) for s in (gap_end - 0.45, gap_end - 0.30, gap_end - 0.15))
        first_speech = win_rms(gap_end, gap_end + 0.15)
        self.assertGreater(peak_gap, 0.0, "停顿里必须有可测的气息能量")
        drop_db = 20 * torch.log10(torch.tensor(first_speech / peak_gap)).item()
        self.assertLess(drop_db, -6.0,
                        f"开口后 150ms 的气息只比停顿峰值低 {-drop_db:.1f} dB，弧没在停顿内走完")


if __name__ == "__main__":
    unittest.main()
