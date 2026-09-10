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
        pitch, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu")
        assert pitch.shape == (3,)
        assert pitchf.shape == (3,)
        assert pitch.dtype == torch.long

    def test_pitch_shift_ignored(self):
        """f0_up_key 已弃用，音域映射始终生效，f0_up_key 不影响输出。"""
        from unittest.mock import patch
        f0 = torch.tensor([200.0])
        with patch("rvc.inference.f0_extractor.experimental_config") as mock_cfg:
            mock_cfg.pitch_map_src_min = 100.0
            mock_cfg.pitch_map_src_max = 500.0
            mock_cfg.pitch_map_dst_min = 100.0
            mock_cfg.pitch_map_dst_max = 500.0
            pitch_0, _ = postprocess_f0(f0, f0_up_key=0, device="cpu")
            pitch_12, _ = postprocess_f0(f0, f0_up_key=12, device="cpu")
        # identity mapping + f0_up_key 被忽略 → 输出相同
        assert pitch_0.item() == pitch_12.item()



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
