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


if __name__ == "__main__":
    unittest.main()
