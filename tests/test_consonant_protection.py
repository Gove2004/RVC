"""清辅音保护测试 — 输出侧混合原始输入音频（effects.py AudioProcessor）。

清辅音保护已从特征侧移到输出侧：对 F0 低于阈值的清音帧，
用原始输入音频替换合成输出，避免 /s/ /f/ /sh/ 等清辅音被过度转换。
"""
import unittest

import torch

from rvc.audio.effects import AudioProcessor


class TestConsonantProtectionOutputSide(unittest.TestCase):
    """输出侧清辅音保护测试。"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.processor.sr = 48000
        self.device = "cpu"

    def test_protect_disabled_at_0(self):
        """protect=0 时关闭保护，输出与合成输入一致。"""
        infer = torch.randn(4800)  # 100ms @ 48kHz
        ref = torch.randn(4800)
        pitchf = torch.tensor([0.0, 100.0, 0.0, 200.0, 0.0, 150.0, 0.0, 180.0, 0.0, 120.0])
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=0.0, protect_threshold_hz=25.0,
        )
        self.assertTrue(torch.allclose(result, infer, atol=1e-6))

    def test_unvoiced_frames_mixed(self):
        """F0 < threshold 的清音帧混合原始输入。"""
        infer = torch.ones(4800) * 0.5  # 合成输出全 0.5
        ref = torch.ones(4800) * 1.0    # 原始输入全 1.0
        # 10 帧 F0，交替清音/浊音
        pitchf = torch.tensor([0.0, 100.0, 0.0, 200.0, 150.0, 0.0, 100.0, 0.0, 200.0, 150.0])
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=1.0, protect_threshold_hz=25.0,
        )
        # 清音帧（第 0, 2, 5, 7 帧）应该接近 1.0（原始输入）
        # 每帧 480 样本，取每帧中间样本验证
        self.assertAlmostEqual(result[240].item(), 1.0, places=2)   # 第 0 帧中间
        self.assertAlmostEqual(result[1200].item(), 1.0, places=2)  # 第 2 帧中间
        self.assertAlmostEqual(result[2640].item(), 1.0, places=2)  # 第 5 帧中间
        self.assertAlmostEqual(result[3600].item(), 1.0, places=2)  # 第 7 帧中间

    def test_voiced_frames_unchanged(self):
        """浊音帧（F0 >= threshold）不受保护影响。"""
        infer = torch.randn(4800)
        ref = torch.randn(4800)
        pitchf = torch.ones(10) * 100.0  # 全浊音
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=1.0, protect_threshold_hz=25.0,
        )
        self.assertTrue(torch.allclose(result, infer, atol=1e-4))

    def test_partial_protection_strength(self):
        """protect=0.5 时清音帧混合一半原始输入。"""
        infer = torch.ones(4800) * 0.0  # 合成输出全 0
        ref = torch.ones(4800) * 1.0    # 原始输入全 1
        pitchf = torch.tensor([0.0, 100.0, 0.0, 200.0, 150.0, 0.0, 100.0, 0.0, 200.0, 150.0])
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=0.5, protect_threshold_hz=25.0,
        )
        # strength = 0.5，清音帧 = 0.5*0 + 0.5*1 = 0.5
        self.assertAlmostEqual(result[240].item(), 0.5, places=2)
        # 浊音帧不受影响，还是 0
        self.assertAlmostEqual(result[720].item(), 0.0, places=2)

    def test_no_pitchf_no_protection(self):
        """pitchf=None 时不启用保护。"""
        infer = torch.randn(4800)
        ref = torch.randn(4800)
        result = self.processor._apply_consonant_protection(
            infer, ref, None, protect=1.0, protect_threshold_hz=25.0,
        )
        self.assertTrue(torch.allclose(result, infer, atol=1e-6))

    def test_empty_pitchf_no_protection(self):
        """pitchf 为空时不启用保护。"""
        infer = torch.randn(4800)
        ref = torch.randn(4800)
        pitchf = torch.tensor([])
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=1.0, protect_threshold_hz=25.0,
        )
        self.assertTrue(torch.allclose(result, infer, atol=1e-6))

    def test_output_shape_preserved(self):
        """保护后输出形状与输入一致。"""
        infer = torch.randn(4800)
        ref = torch.randn(4800)
        pitchf = torch.randn(10).abs() * 100
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=0.5, protect_threshold_hz=25.0,
        )
        self.assertEqual(result.shape, infer.shape)

    def test_ref_shorter_than_infer(self):
        """ref 比 infer 短时自动填充。"""
        infer = torch.ones(4800) * 0.5
        ref = torch.ones(2400) * 1.0  # 只有一半长度
        pitchf = torch.zeros(10)  # 全清音
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=1.0, protect_threshold_hz=25.0,
        )
        # 前半部分应该是 1.0（ref 有值），后半部分应该是 0（ref 填充 0）
        self.assertAlmostEqual(result[1000].item(), 1.0, places=2)
        self.assertAlmostEqual(result[3000].item(), 0.0, places=2)

    def test_threshold_boundary(self):
        """F0 恰好等于阈值时视为浊音（不保护）。"""
        infer = torch.ones(4800) * 0.5
        ref = torch.ones(4800) * 1.0
        pitchf = torch.ones(10) * 25.0  # 恰好等于阈值
        result = self.processor._apply_consonant_protection(
            infer, ref, pitchf, protect=1.0, protect_threshold_hz=25.0,
        )
        # F0 >= threshold 视为浊音，不保护
        self.assertTrue(torch.allclose(result, infer, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
