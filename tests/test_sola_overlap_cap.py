"""SOLA 重叠上限契约测试。

背景：commit 6153bfd 把 SOLA 重叠上限从 4*zc（40ms）放宽到 5*zc（50ms），
同时把 crossfade_time 缺省值 0.04→0.05、GUI 滑块上限 0.04→0.05。三处联动
的结果是用户实际听到的重叠从 40ms 变成 50ms，听感退化为「每个字持续时间
太短、听不清」。

机制：重叠区里旧块尾部（fade_out）与新块头部（fade_in）按 sola_offset
错位相加，错位量可达整个搜索窗。错位相加造成周期性相消，该区段能量下沉。
重叠越长，被相消区替换掉的原始音频越多——块长 100ms 时 50ms 重叠意味着
半个块都在相消区里。放宽上限没有换来任何对齐收益，因此上限必须锁死在
40ms 且**与块长无关**（与上游 RVCRealtimeVST 的 `min(crossfade_frame, 4*zc)`
一致）。

本测试锁定三处联动，任何一处再被单独放宽都会失败。
"""
import unittest

from rvc.core.config import BufferParams
from rvc.streaming.runner import InferenceRunner


class TestSolaOverlapCap(unittest.TestCase):
    """重叠上限 = min(crossfade_samples, 4*zc)，与块长无关。"""

    def _sola_samples(self, work_sr, block_time, crossfade_time):
        """复现 InferenceRunner.init_processing 的参数计算（不建模型）。"""
        import numpy as np
        zc = work_sr // 100
        block_samples = int(np.round(block_time * work_sr / zc)) * zc
        crossfade_samples = int(np.round(crossfade_time * work_sr / zc)) * zc
        sola_buffer_samples = min(crossfade_samples, 4 * zc)
        return zc, block_samples, crossfade_samples, sola_buffer_samples

    def test_cap_is_40ms_at_40k(self):
        zc, _, _, sola = self._sola_samples(40000, 0.1, 0.05)
        self.assertEqual(zc, 400)
        self.assertEqual(sola, 4 * zc)
        self.assertEqual(sola, 1600)
        self.assertAlmostEqual(sola / 40000, 0.04)

    def test_cap_is_40ms_at_48k(self):
        zc, _, _, sola = self._sola_samples(48000, 0.25, 0.05)
        self.assertEqual(zc, 480)
        self.assertEqual(sola, 4 * zc)
        self.assertAlmostEqual(sola / 48000, 0.04)

    def test_cap_independent_of_block_length(self):
        """上游注释明确要求：重叠与回调块长无关。"""
        results = [
            self._sola_samples(40000, bt, 0.05)[3]
            for bt in (0.05, 0.1, 0.25, 0.5, 1.0)
        ]
        self.assertEqual(len(set(results)), 1, f"重叠随块长变化: {results}")
        self.assertEqual(results[0], 1600)

    def test_short_crossfade_is_not_clamped(self):
        """crossfade_time 小于上限时按配置走，不被 40ms 顶上去。"""
        zc, _, cf, sola = self._sola_samples(40000, 0.1, 0.01)
        self.assertEqual(cf, 400)
        self.assertEqual(sola, 400)

    def test_overlap_never_exceeds_block(self):
        """重叠 + 搜索窗必须塞得进一个块，否则 OLA 会吃掉真实音频。"""
        for bt in (0.05, 0.1, 0.25, 0.5, 1.0):
            zc, block, _, sola = self._sola_samples(40000, bt, 0.05)
            self.assertLessEqual(
                sola + zc, block,
                f"block_time={bt}: 重叠{sola}+搜索{zc} > 块长{block}",
            )


class TestCrossfadeDefaultsAgree(unittest.TestCase):
    """缺省值、GUI 滑块上限、SOLA 上限三处必须一致，否则用户拿到的不是 40ms。"""

    def test_config_default_is_40ms(self):
        self.assertAlmostEqual(BufferParams().crossfade_time, 0.04)

    def test_config_default_equals_sola_cap(self):
        zc = 400
        cap_seconds = 4 * zc / 40000
        self.assertAlmostEqual(BufferParams().crossfade_time, cap_seconds)

    def test_gui_slider_max_matches_config_default(self):
        """滑块上限即 SOLA 上限；改一处必须同步另一处。"""
        import re
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "gui" / "view" / "tabs" / "params_tab.py"
        text = src.read_text(encoding="utf-8")
        m = re.search(
            r"crossfade_slider,\s*crossfade_label\s*=\s*create_slider_row\(\s*"
            r"([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
            text,
        )
        self.assertIsNotNone(m, "未找到 crossfade 滑块定义")
        lo, hi, step, default = (float(m.group(i)) for i in range(1, 5))
        self.assertAlmostEqual(hi, 0.04, msg="滑块上限必须等于 SOLA 上限 40ms")
        self.assertAlmostEqual(default, 0.04)
        self.assertLessEqual(lo, default)
        self.assertGreater(step, 0)


class TestSolaCapIsNotFiveZc(unittest.TestCase):
    """直接盯住源码，防止 5*zc 回归。"""

    def test_runner_uses_four_zc(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "rvc" / "streaming" / "runner.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("min(state.crossfade_samples, 4 * zc)", text)
        self.assertNotIn("min(state.crossfade_samples, 5 * zc)", text)


if __name__ == "__main__":
    unittest.main()
