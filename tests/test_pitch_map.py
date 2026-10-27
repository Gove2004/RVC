"""rvc.pipeline.pitch.postprocess.apply_pitch_map — 音域映射核心语义。

关键契约（用户裁定，勿动）：
- 半音尺度线性映射（音程按常数斜率 k 缩放，k<1 时压缩——刻意行为）
- 界外钳制到模型音域边界（不静音）
- 清音帧（0）保持 0
"""
import math
import unittest

import torch

from rvc.dsp.hz_midi import hz_to_midi
from rvc.pipeline.pitch.postprocess import apply_pitch_map


class TestPitchMap(unittest.TestCase):
    def test_identity_when_domains_equal(self):
        f0 = torch.tensor([80.0, 200.0, 400.0, 600.0])
        out = apply_pitch_map(f0, 80.0, 640.0, 80.0, 640.0)
        self.assertTrue(torch.allclose(out, f0, rtol=1e-4))

    def test_interval_scaled_by_constant_slope(self):
        # 设计行为：半音尺度线性映射，音程按常数斜率 k 缩放（k<1 时压缩——
        # 用户明确裁定保留该效果，勿以修 bug 名义改动）
        f0 = torch.tensor([200.0, 400.0])  # 相距 12 半音
        out = apply_pitch_map(f0, 40.0, 640.0, 120.0, 720.0)
        dist_in = 12.0
        dist_out = math.log2(float(out[1]) / float(out[0])) * 12
        k = (float(hz_to_midi(torch.tensor(720.0))) - float(hz_to_midi(torch.tensor(120.0)))) / (
            float(hz_to_midi(torch.tensor(640.0))) - float(hz_to_midi(torch.tensor(40.0))))
        self.assertAlmostEqual(dist_out, dist_in * k, places=3)

    def test_clamp_above_dst_max(self):
        # >640Hz 全部钳到 720（保持有声，不变静音）
        f0 = torch.tensor([650.0, 800.0, 1000.0])
        out = apply_pitch_map(f0, 40.0, 640.0, 120.0, 720.0)
        for v in out:
            self.assertAlmostEqual(float(v), 720.0, places=1)

    def test_clamp_below_dst_min(self):
        f0 = torch.tensor([10.0, 20.0])
        out = apply_pitch_map(f0, 40.0, 640.0, 120.0, 720.0)
        for v in out:
            self.assertAlmostEqual(float(v), 120.0, places=1)

    def test_uv_frames_stay_zero(self):
        f0 = torch.tensor([0.0, 200.0, 0.0])
        out = apply_pitch_map(f0, 40.0, 640.0, 120.0, 720.0)
        self.assertEqual(float(out[0]), 0.0)
        self.assertEqual(float(out[2]), 0.0)

    def test_invalid_params_return_original(self):
        f0 = torch.tensor([200.0])
        out = apply_pitch_map(f0, 640.0, 40.0, 120.0, 720.0)  # src 倒置
        self.assertTrue(torch.equal(out, f0))


class TestCoarseF0MatchesUpstream(unittest.TestCase):
    """离散 pitch 的 bin 划分必须与上游 RVC 逐值一致。

    背景：`f0_mel_min/max` 的取值很容易被"凭记忆"误判成上游硬编码的 112.0/1975.0
    （网上流传的说法），从而被当成 bug 去"修正"。实际上游源码
    （`train/dataset/extract_f0.py` 与 `infer/rtrvc.py::get_f0_post`）写的是：

        f0_min, f0_max = 50.0, 1100.0
        f0_mel_min = 1127 * log(1 + f0_min / 700)
        f0_mel_max = 1127 * log(1 + f0_max / 700)
        f0_mel = 1127 * log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = round(f0_mel)

    与本项目的 normalize_f0_to_coarse 等价。本类把上游公式独立重写一份做对照，
    一旦有人"修正"常量就会失败。注意模型是从上游 f0G40k.pth 微调的，
    bin 划分错位不会让音高明显跑调（输出音高由连续 nsff0 决定），
    只会让 enc_p 收到训练时没见过的音高条件 —— 表现为咬字/音色发闷，
    非常难归因，所以必须用测试钉死。
    """

    F0_MIN = 50.0
    F0_MAX = 1100.0

    @classmethod
    def _upstream_reference(cls, f0):
        """逐字照抄上游公式的独立实现（numpy）。"""
        import numpy as np

        f0 = np.asarray(f0, dtype=np.float64).copy()
        mel_min = 1127 * math.log(1 + cls.F0_MIN / 700)
        mel_max = 1127 * math.log(1 + cls.F0_MAX / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        mask = f0_mel > 0
        f0_mel[mask] = (f0_mel[mask] - mel_min) * 254 / (mel_max - mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int64)

    def test_mel_bounds_use_f0_formula(self):
        """常量必须是公式算出来的，不能是 112.0/1975.0。"""
        from rvc.dsp.mel import F0_MEL_MAX, F0_MEL_MIN

        self.assertAlmostEqual(F0_MEL_MIN, 1127 * math.log(1 + 50.0 / 700), places=9)
        self.assertAlmostEqual(F0_MEL_MAX, 1127 * math.log(1 + 1100.0 / 700), places=9)
        self.assertNotAlmostEqual(F0_MEL_MIN, 112.0, places=1)
        self.assertNotAlmostEqual(F0_MEL_MAX, 1975.0, places=1)

    def test_bins_match_upstream_across_range(self):
        """50~1100Hz 全量程逐值对照上游。"""
        import numpy as np

        from rvc.pipeline.pitch.postprocess import normalize_f0_to_coarse

        f0 = np.linspace(1.0, 1200.0, 1200)
        mine = normalize_f0_to_coarse(f0.astype(np.float32))
        ref = self._upstream_reference(f0)
        np.testing.assert_array_equal(
            np.asarray(mine), ref,
            err_msg="离散 pitch 与上游不一致：bin 划分被改动了")

    def test_uv_maps_to_one(self):
        """清音帧（f0=0）上游给 1，本项目 where(...,0)+clip(下限1) 也应是 1。"""
        import numpy as np

        from rvc.pipeline.pitch.postprocess import normalize_f0_to_coarse

        out = normalize_f0_to_coarse(np.array([0.0, 0.0], dtype=np.float32))
        self.assertEqual(list(np.asarray(out)), [1, 1])


if __name__ == "__main__":
    unittest.main()
