"""F0 提取器抽象接口单元测试。"""
import unittest
import torch

from rvc.audio.f0_utils import normalize_f0_to_coarse

from rvc.inference.f0_extractor import (
    F0Extractor,
    RMVPEExtractor,
    FCPEExtractor,
    postprocess_f0,
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
            def extract(self, audio, sr, f0_up_key):
                pass

        with self.assertRaises(TypeError):
            IncompleteExtractor()


class TestPostprocessF0(unittest.TestCase):
    """postprocess_f0 函数测试。"""

    def test_basic(self):
        f0 = torch.tensor([100.0, 200.0, 0.0])
        pitch, pitchf = postprocess_f0(f0, device="cpu")
        assert pitch.shape == (3,)
        assert pitchf.shape == (3,)
        assert pitch.dtype == torch.long



class TestNormalizeF0(unittest.TestCase):
    """_normalize_f0_to_coarse 函数测试。"""

    def test_range(self):
        f0 = torch.tensor([50.0, 100.0, 1000.0, 2000.0])
        result = normalize_f0_to_coarse(f0)
        assert result.dtype == torch.long
        assert torch.all(result >= 1)
        assert torch.all(result <= 255)

    def test_zero_f0(self):
        """F0=0（清音）应映射到 1。"""
        f0 = torch.tensor([0.0])
        result = normalize_f0_to_coarse(f0)
        assert result[0] == 1
