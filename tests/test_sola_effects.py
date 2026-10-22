"""SOLA 拼接与输出效果链 — CPU 单元测试。

锁定的核心契约：
- apply_sola 输出恒为 block_samples 定长（SOLA 对齐契约）
- 平滑周期信号的连续块经 SOLA 拼接后无跳变（不产生咔哒/重影）
- NaN 防护：buffer 污染后输出仍有限值，buffer 复位为零
- RmsMixEffect：rms_mix=1 恒等旁路；rms_mix=0 跟随参考响度
- AudioProcessor 全链可在 CPU 上 setup/process 往返
"""
import unittest

import numpy as np
import torch

from rvc.dsp.sola import apply_sola
from rvc.streaming.effects import AudioProcessor, RmsMixEffect, SolaEffect

BLOCK = 1000
BUF = 100
SEARCH = 100


def _make_sola_effect() -> SolaEffect:
    eff = SolaEffect()
    eff.setup(sr=100, block_samples=BLOCK, sola_buffer_samples=BUF,
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


if __name__ == "__main__":
    unittest.main()
