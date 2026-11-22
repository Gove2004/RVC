"""SOLA 拼接与输出效果链 — CPU 单元测试。

锁定的核心契约：
- apply_sola 输出恒为 block_samples 定长（SOLA 对齐契约）
- 平滑周期信号的连续块经 SOLA 拼接后无跳变（不产生咔哒/重影）
- 交叉淡化长于块长（buf > block）时稳态仍逐样本还原输入
- NaN 防护：buffer 污染后输出仍有限值，buffer 复位为零
- RmsMixEffect：rms_mix=1 恒等旁路；rms_mix=0 跟随参考响度
- MicAirflowEffect：强度 0 恒等旁路；输出不随分块长度漂移（ulp 级）；强度单调；
  起弧以零斜率离开保持段（无拐点）
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


class TestApplySolaBufferLongerThanBlock(unittest.TestCase):
    """buf > block 的良性证明：嵌套淡入被互补抵消，稳态逐样本还原输入。

    buf 存的 tail 在落盘前已被原地淡入，但下一块混合时它的搭档是同一段
    音频的另一次合成，fade_in + fade_out ≡ 1 会精确抵消；输出又只消费
    buffer[:block]。若以后把 SOLA 换成归一化 OLA 或改 buffer 语义，这里
    会立刻失败——届时必须同步约束 crossfade_time ≤ block_time。
    """

    def test_steady_state_restores_input(self):
        block, buf, search = 1000, 2000, 100
        n_blocks = 10
        total = n_blocks * block + buf + search
        t = np.arange(total, dtype=np.float32)
        # 三段不可通约的正弦叠加：非周期，否则重复段同相会掩盖增益偏差
        x = (0.5 * np.sin(2 * np.pi * t / 173)
             + 0.3 * np.sin(2 * np.pi * t / 419)
             + 0.2 * np.sin(2 * np.pi * t / 997)).astype(np.float32)
        eff = SolaEffect()
        eff.setup(block_samples=block, sola_buffer_samples=buf,
                  sola_search_samples=search, device="cpu")
        outs = []
        for i in range(n_blocks):
            seg = torch.from_numpy(x[i * block:i * block + block + buf + search].copy())
            outs.append(eff.process(seg).numpy())
        joined = np.concatenate(outs)
        # 启动淡入时长 = crossfade（与 block 无关），跳过 buf 样本再看稳态
        np.testing.assert_allclose(
            joined[buf:], x[:joined.size][buf:], atol=1e-5, rtol=0,
            err_msg="buf > block 时稳态拼接不再还原输入（应恒等，容差 1e-5）",
        )


class TestApplySolaLengthContract(unittest.TestCase):
    """apply_sola 的长度契约：输入必须 >= block + buffer + search。

    不足时 tail 为空张量，`sola_buffer[:] = safe_tail` 会抛尺寸错，且**是否崩
    取决于 argmax 选中的 offset**——是概率性的、难复现的崩溃（实测 L=need-1
    时约 4/200 概率、L=need-3 时 19/20 成功）。故在入口显式校验，
    与 rvc/dsp/rms.py 的 fast_rms 同风格。
    """

    def _args(self, block, buf, search):
        return dict(
            sola_buffer=torch.zeros(buf),
            sola_norm_kernel=torch.ones(1, 1, buf),
            fade_in=torch.ones(buf),
            fade_out=torch.ones(buf),
            block_samples=block,
            sola_buffer_samples=buf,
            sola_search_samples=search,
        )

    def test_short_input_raises_valueerror(self):
        block, buf, search = 2000, 4000, 400
        need = block + buf + search
        for length in (0, 1, need - 1000, need - 1):
            with self.subTest(length=length):
                with self.assertRaises(ValueError) as ctx:
                    apply_sola(torch.randn(length), **self._args(block, buf, search))
                # 报错必须点明期望长度与三项组成，否则等于没报
                msg = str(ctx.exception)
                self.assertIn(str(need), msg)
                self.assertIn(str(block), msg)
                self.assertIn(str(buf), msg)
                self.assertIn(str(search), msg)

    def test_short_input_is_deterministic_not_flaky(self):
        """同一非法长度必须**每次**都报错（修复前是概率性崩溃）。"""
        block, buf, search = 2000, 4000, 400
        need = block + buf + search
        for _ in range(20):
            with self.assertRaises(ValueError):
                apply_sola(torch.randn(need - 1), **self._args(block, buf, search))

    def test_exact_length_is_accepted(self):
        """边界：恰好等于 need 必须正常返回 block 长输出。"""
        block, buf, search = 2000, 4000, 400
        out = apply_sola(torch.randn(block + buf + search),
                         **self._args(block, buf, search))
        self.assertEqual(tuple(out.shape), (block,))


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

    def test_air_is_voice_correlated_and_dies_with_the_voice(self):
        """说话中的气息必须跟着语音走，不是独立底噪。

        只开关说话气息层前后相减（吸气/底噪两次都关掉），得到纯气息分量：
        元音里要有气息；元音结束后 300ms 要明显衰减（旧版独立噪声床的尾音
        嘶声正是被这条否决的）；满滑条也不得接近语音电平。
        """
        sr = self.SR
        t = torch.arange(int(1.5 * sr), dtype=torch.float32) / sr
        x = torch.cat([0.3 * torch.sin(2 * torch.pi * 220 * t),
                       torch.zeros(int(0.5 * sr))])
        saved = (MicAirflowEffect.BREATH_GAIN, MicAirflowEffect.FLOOR_GAIN,
                 MicAirflowEffect.AIR_GAIN)
        try:
            MicAirflowEffect.BREATH_GAIN = MicAirflowEffect.FLOOR_GAIN = 0.0
            full = self._effect().process(x, 1.0)
            MicAirflowEffect.AIR_GAIN = 0.0
            bare = self._effect().process(x, 1.0)
        finally:
            (MicAirflowEffect.BREATH_GAIN, MicAirflowEffect.FLOOR_GAIN,
             MicAirflowEffect.AIR_GAIN) = saved
        noise = (full - bare).pow(2)
        speech_rms = float(x.pow(2).mean().sqrt())

        def rms_db(a_s, b_s):
            v = float(noise[int(a_s * sr):int(b_s * sr)].mean().sqrt())
            return 20 * np.log10(max(v, 1e-9) / speech_rms), v

        in_voice = rms_db(1.0, 1.5)[0]
        after = rms_db(1.5 + 0.30, 1.5 + 0.45)[0]
        self.assertGreater(in_voice, -40.0, f"说话中气息 {in_voice:.1f} dB，几乎听不到")
        self.assertLess(in_voice, -8.0, f"说话中气息 {in_voice:.1f} dB，快追上语音了")
        self.assertLess(after, in_voice - 8.0,
                        f"元音结束后 300ms 气息只掉了 {in_voice - after:.1f} dB，"
                        "像独立噪声床的尾音")

    def test_breath_depth_grows_with_preceding_speech(self):
        """气量：同样的 0.8s 停顿，前面句子越长，吸气越深（短句只换到浅吸）。"""
        sr = self.SR

        def speech(dur):
            t = torch.arange(int(dur * sr), dtype=torch.float32) / sr
            return 0.3 * torch.sin(2 * torch.pi * 220 * t)

        gap = torch.full((int(0.8 * sr),), 1e-4, dtype=torch.float32)
        x = torch.cat([speech(0.4), gap, speech(2.0), gap])
        y = self._effect().process(x, 1.0)
        add = (y - x).pow(2)

        def gap_peak(start_s):
            a, b = int(start_s * sr), int((start_s + 0.8) * sr)
            return float(add[a:b].mean().sqrt())

        short_peak = gap_peak(0.4)
        long_peak = gap_peak(0.4 + 0.8 + 2.0)
        self.assertGreater(long_peak, 1.5 * short_peak,
                           f"长句后的吸气 {long_peak:.5f} 不比短句后 {short_peak:.5f} 深")

    def test_breath_never_dominates_the_voice(self):
        """停顿里的气声不得盖过人声（满滑条、长句后的深吸是最响的情形）。

        上界取语音 RMS 的 1.8 倍（+5dB）：中段滑条远低于语音，只有「长句 +
        满滑条 + 连续发力」三者同时成立才逼近这个上限——回归防线，不是设计目标。
        """
        sr = self.SR
        t = torch.arange(int(3.0 * sr), dtype=torch.float32) / sr
        speech = 0.3 * torch.sin(2 * torch.pi * 220 * t)
        gap = torch.full((int(0.8 * sr),), 1e-4, dtype=torch.float32)
        x = torch.cat([speech, gap])
        y = self._effect().process(x, 1.0)
        # 50ms 窗 RMS：宽带噪声的逐样本峰值没有听感意义
        add = (y - x)[int(3.0 * sr):]
        w = sr // 20
        peak = float(add[:add.shape[0] // w * w].pow(2).reshape(-1, w).mean(1).max().sqrt())
        speech_rms = float(speech.pow(2).mean().sqrt())
        self.assertLess(peak, 1.8 * speech_rms,
                        f"50ms 吸气峰值 {peak:.4f} 超过语音 {speech_rms:.4f} 的 1.8 倍，气声太响")

    def test_arc_onset_leaves_zero_without_slope_corner(self):
        """起弧必须以零斜率离开 0：保持段后的第一段斜率远小于弧内最大斜率。

        旧写法在 0.15s 保持段结束时把上升速率从 0 硬切到 α，而一阶弧的斜率
        最大值恰在起点，包络因此带拐点，听感是停顿后「呼」地冒出。这里锁形状
        类别（首段缓慢、随后加速），不锁具体常数。
        """
        sr, eff = self.SR, self._effect()
        voiced = 0.3 * torch.sin(
            2 * torch.pi * 220 * torch.arange(int(0.6 * sr), dtype=torch.float32) / sr)
        x = torch.cat([voiced, torch.zeros(int(1.0 * sr)), voiced])
        step = int(0.01 * sr)
        envs = []
        for i in range(0, x.shape[0], step):
            eff.process(x[i:i + step], 1.0)
            envs.append(float(eff._env_tail))
        gap = np.array(envs)[int(0.6 * sr) // step:]
        start = int(np.argmax(gap > 0))
        d = np.diff(gap)
        self.assertLess(d[start], 0.2 * d.max(),
                        f"起弧速率 {d[start]:.4f} 已接近弧内最大 {d.max():.4f}"
                        "——包络以拐点离开 0")
        self.assertLess(gap[start + 5] / gap.max(), 0.2,
                        f"起弧 50ms 已到峰值 {gap[start + 5] / gap.max():.0%}，"
                        "前半段上升太快")

    def test_strength_is_monotonic_and_output_finite(self):
        """滑条越强，停顿里加的气声越多。

        只看静音段：语音段 (out-x) 被静态饱和压峰的负贡献主导（近讲提升随强度
        增加输入幅度，压缩损失按强度非线性增长），全频段能量并不反映气声强弱。
        """
        x = _speech_with_pauses(40000, self.SR)
        quiet = x.abs() < 1e-3
        deltas = []
        for strength in (0.3, 0.7, 1.0):
            out = self._effect().process(x, strength)
            self.assertTrue(torch.isfinite(out).all())
            deltas.append(float((out - x)[quiet].pow(2).mean().sqrt()))
        self.assertTrue(deltas[0] < deltas[1] < deltas[2], f"强度应单调递增：{deltas}")

    def test_inhale_arc_fades_before_speech_resumes(self):
        """吸气弧必须在停顿内部走完：开口瞬间的气息能量要明显低于停顿里的峰值。

        反例是 git 版的「升到开口那一刻才到顶」——呼呼压在每个停顿后的第一个字上
        再被 τ=0.05s 硬切，实测断句处比停顿峰值只低 0.6 dB（听感=位置错+突兀）。
        """
        sr, eff = self.SR, self._effect()
        # 2.5s 句子：气量积分到 ~0.8，事件电平高于静音底噪，比例才测得到弧
        voiced = 0.3 * torch.sin(
            2 * torch.pi * 220 * torch.arange(int(2.5 * sr), dtype=torch.float32) / sr)
        gap = torch.full((int(0.8 * sr),), 1e-4, dtype=torch.float32)
        x = torch.cat([voiced, gap, voiced])
        y = eff.process(x, 1.0)
        ins = (y - x).pow(2)

        def win_rms(a, b):
            return float(ins[int(a * sr):int(b * sr)].mean().sqrt())

        gap_end = 2.5 + 0.8
        peak_gap = max(win_rms(s, s + 0.15) for s in (gap_end - 0.45, gap_end - 0.30, gap_end - 0.15))
        first_speech = win_rms(gap_end, gap_end + 0.15)
        self.assertGreater(peak_gap, 0.0, "停顿里必须有可测的气息能量")
        drop_db = 20 * torch.log10(torch.tensor(first_speech / peak_gap)).item()
        self.assertLess(drop_db, -6.0,
                        f"开口后 150ms 的气息只比停顿峰值低 {-drop_db:.1f} dB，弧没在停顿内走完")


if __name__ == "__main__":
    unittest.main()
