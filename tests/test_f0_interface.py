"""F0 提取器抽象接口单元测试。"""
import unittest
import torch

from rvc.inference.f0_extractor import (
    F0Extractor,
    RMVPEExtractor,
    FCPEExtractor,
    postprocess_f0,
    apply_f0_break_protect,
    _normalize_f0_to_coarse,
)


class TestF0ExtractorABC(unittest.TestCase):
    """F0Extractor 抽象基类测试。"""

    def test_cannot_instantiate(self):
        """抽象基类不能直接实例化。"""
        with self.assertRaises(TypeError):
            F0Extractor()

    def test_subclass_must_implement_extract(self):
        """子类必须实现 extract 方法。"""
        class IncompleteExtractor(F0Extractor):
            def clear_cuda_graph(self):
                pass

        with self.assertRaises(TypeError):
            IncompleteExtractor()

    def test_subclass_must_implement_clear_cuda_graph(self):
        """子类必须实现 clear_cuda_graph 方法。"""
        class IncompleteExtractor(F0Extractor):
            def extract(self, audio, sr, f0_up_key, f0_proc=None):
                pass

        with self.assertRaises(TypeError):
            IncompleteExtractor()


class TestPostprocessF0(unittest.TestCase):
    """postprocess_f0 函数测试。"""

    def test_basic(self):
        f0 = torch.tensor([100.0, 200.0, 0.0])
        pitch, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu", f0_proc=None)
        assert pitch.shape == (3,)
        assert pitchf.shape == (3,)
        assert pitch.dtype == torch.long

    def test_pitch_shift(self):
        f0 = torch.tensor([100.0])
        pitch_0, _ = postprocess_f0(f0, f0_up_key=0, device="cpu", f0_proc=None)
        pitch_12, _ = postprocess_f0(f0, f0_up_key=12, device="cpu", f0_proc=None)
        # 升高 12 半音，F0 应翻倍，离散 pitch 应增大
        assert pitch_12.item() > pitch_0.item()


class TestBreakProtect(unittest.TestCase):
    """破音保护函数测试。"""

    def test_disabled_ratio_1(self):
        """ratio=1 时不压缩。"""
        f0 = torch.tensor([100.0, 500.0, 1000.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=1.0)
        assert torch.allclose(result, f0)

    def test_below_critical_unchanged(self):
        """低于临界值不压缩。"""
        f0 = torch.tensor([100.0, 200.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.5)
        assert torch.allclose(result, f0)

    def test_above_critical_compressed(self):
        """高于临界值+膝宽应压缩。"""
        f0 = torch.tensor([1000.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.5, knee=0.0)
        # 压缩后应小于原值
        assert result[0] < f0[0]

    def test_zero_critical_no_op(self):
        """critical_hz<=0 时原样返回。"""
        f0 = torch.tensor([100.0, 200.0])
        result = apply_f0_break_protect(f0, critical_hz=0.0)
        assert torch.allclose(result, f0)


class TestNormalizeF0(unittest.TestCase):
    """_normalize_f0_to_coarse 函数测试。"""

    def test_range(self):
        f0 = torch.tensor([50.0, 100.0, 1000.0, 2000.0])
        result = _normalize_f0_to_coarse(f0)
        assert result.dtype == torch.long
        assert torch.all(result >= 1)
        assert torch.all(result <= 255)

    def test_zero_f0(self):
        """F0=0（清音）应映射到 1。"""
        f0 = torch.tensor([0.0])
        result = _normalize_f0_to_coarse(f0)
        assert result[0] == 1
