"""F0 extractor deep tests - pure functions only.

Covers:
- _FilteredStream: prefix/contains filtering, empty text, flush, __getattr__
- _normalize_f0_to_coarse: various F0 values, boundaries, UV
- postprocess_f0: pitch map (semitone scale), numpy/tensor input
- constants: FCPE_CONFIDENCE_THRESHOLD
- create_f0_extractor: unknown method raises
"""
import io
import sys
import unittest

import torch

from rvc.inference.f0_extractor import (
    FCPE_CONFIDENCE_THRESHOLD,
    F0Extractor,
    _FilteredStream,
    _suppress_third_party_output,
    create_f0_extractor,
    postprocess_f0,
)
from rvc.audio.f0_utils import normalize_f0_to_coarse
from rvc.core.errors import F0ExtractionError


class TestFilteredStream(unittest.TestCase):
    """Tests for _FilteredStream."""

    def setUp(self):
        self.buffer = io.StringIO()
        self.stream = _FilteredStream(
            self.buffer,
            blocked_prefixes=("[INFO]", "[WARN]"),
            blocked_contains=("error",),
        )

    def test_pass_through_normal_text(self):
        self.stream.write("hello world\n")
        self.assertIn("hello world", self.buffer.getvalue())

    def test_block_prefix(self):
        self.stream.write("[INFO] this should be blocked\n")
        self.assertEqual(self.buffer.getvalue(), "")

    def test_block_prefix_with_leading_spaces(self):
        self.stream.write("   [WARN] this should be blocked\n")
        self.assertEqual(self.buffer.getvalue(), "")

    def test_block_contains(self):
        self.stream.write("there is an error here\n")
        self.assertEqual(self.buffer.getvalue(), "")

    def test_empty_text(self):
        self.stream.write("")
        self.assertEqual(self.buffer.getvalue(), "")

    def test_whitespace_only(self):
        self.stream.write("   \n")
        self.assertEqual(self.buffer.getvalue(), "")

    def test_returns_length(self):
        """write returns number of chars written even if filtered."""
        n = self.stream.write("[INFO] blocked\n")
        self.assertEqual(n, len("[INFO] blocked\n"))

    def test_flush(self):
        self.stream.write("test\n")
        self.stream.flush()  # should not crash

    def test_getattr_delegates(self):
        """__getattr__ delegates to underlying stream."""
        self.stream.write("test\n")
        self.assertIn("test", self.stream.getvalue())

    def test_multiple_prefixes(self):
        stream = _FilteredStream(
            self.buffer,
            blocked_prefixes=("[A]", "[B]", "[C]"),
            blocked_contains=(),
        )
        stream.write("[A] blocked\n")
        stream.write("[B] blocked\n")
        stream.write("[C] blocked\n")
        stream.write("[D] passed\n")
        self.assertIn("[D] passed", self.buffer.getvalue())
        self.assertNotIn("[A]", self.buffer.getvalue())

    def test_no_prefixes_no_contains(self):
        stream = _FilteredStream(self.buffer, blocked_prefixes=(), blocked_contains=())
        stream.write("anything goes\n")
        self.assertIn("anything goes", self.buffer.getvalue())


class TestSuppressThirdPartyOutput(unittest.TestCase):
    """Tests for _suppress_third_party_output context manager."""

    def test_restores_stdout(self):
        original_stdout = sys.stdout
        with _suppress_third_party_output("[TEST]"):
            self.assertIsNot(sys.stdout, original_stdout)
        self.assertIs(sys.stdout, original_stdout)

    def test_restores_stderr(self):
        original_stderr = sys.stderr
        with _suppress_third_party_output("[TEST]"):
            self.assertIsNot(sys.stderr, original_stderr)
        self.assertIs(sys.stderr, original_stderr)

    def test_suppresses_prefix(self):
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        try:
            with _suppress_third_party_output("[BLOCK]"):
                print("[BLOCK] this should be suppressed")
                print("this should pass")
        finally:
            sys.stdout = original_stdout
        self.assertNotIn("[BLOCK]", buffer.getvalue())
        self.assertIn("this should pass", buffer.getvalue())

    def test_restores_on_exception(self):
        original_stdout = sys.stdout
        try:
            with _suppress_third_party_output("[TEST]"):
                raise ValueError("test exception")
        except ValueError:
            pass
        self.assertIs(sys.stdout, original_stdout)


class TestNormalizeF0ToCoarse(unittest.TestCase):
    """Tests for _normalize_f0_to_coarse."""

    def test_zero_f0_returns_one(self):
        """F0=0 (UV) maps to 1."""
        f0 = torch.tensor([0.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 1)

    def test_negative_f0_returns_one(self):
        """Negative F0 maps to 1."""
        f0 = torch.tensor([-10.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 1)

    def test_very_high_f0_clamped_to_255(self):
        """Very high F0 is clamped to 255."""
        f0 = torch.tensor([10000.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 255)

    def test_normal_f0_in_range(self):
        """Normal F0 (100-1000 Hz) is in 1-255 range."""
        for hz in [100, 200, 300, 440, 500, 800, 1000]:
            f0 = torch.tensor([float(hz)])
            coarse = normalize_f0_to_coarse(f0)
            self.assertGreaterEqual(coarse[0].item(), 1)
            self.assertLessEqual(coarse[0].item(), 255)

    def test_higher_f0_higher_coarse(self):
        """Higher F0 gives higher coarse value (monotonic)."""
        f0_low = torch.tensor([200.0])
        f0_high = torch.tensor([400.0])
        coarse_low = normalize_f0_to_coarse(f0_low)
        coarse_high = normalize_f0_to_coarse(f0_high)
        self.assertGreater(coarse_high[0].item(), coarse_low[0].item())

    def test_output_dtype_long(self):
        f0 = torch.tensor([440.0, 500.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertEqual(coarse.dtype, torch.long)

    def test_batch_input(self):
        f0 = torch.tensor([0.0, 200.0, 440.0, 10000.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertEqual(coarse.shape, (4,))
        self.assertEqual(coarse[0].item(), 1)
        self.assertEqual(coarse[3].item(), 255)

    def test_mid_f0_around_128(self):
        """Mid F0 should be around 128."""
        f0 = torch.tensor([500.0])
        coarse = normalize_f0_to_coarse(f0)
        self.assertGreater(coarse[0].item(), 50)
        self.assertLess(coarse[0].item(), 200)



class TestPostprocessF0(unittest.TestCase):
    """Tests for postprocess_f0（音域映射始终生效）。"""

    def _identity_config(self):
        """mock experimental_config 为 identity mapping（src=dst），f0 不变。"""
        from unittest.mock import patch
        patcher = patch("rvc.inference.f0_extractor.experimental_config")
        mock_cfg = patcher.start()
        mock_cfg.pitch_map_src_min = 100.0
        mock_cfg.pitch_map_src_max = 500.0
        mock_cfg.pitch_map_dst_min = 100.0
        mock_cfg.pitch_map_dst_max = 500.0
        # 清音帧保护相关配置（postprocess_f0 新增）
        mock_cfg.protect_soft_threshold_hz = 20.0
        mock_cfg.protect_soft_width = 30.0
        mock_cfg.rmvpe_threshold = 0.05
        self.addCleanup(patcher.stop)

    def test_identity_mapping_unchanged(self):
        """identity mapping（src=dst）时 pitch 不变。"""
        self._identity_config()
        f0 = torch.tensor([200.0, 300.0])
        coarse, pitchf, _, _ = postprocess_f0(f0, device="cpu")
        self.assertAlmostEqual(pitchf[0].item(), 200.0, delta=1.0)

    def test_numpy_input(self):
        """numpy array input also works."""
        self._identity_config()
        import numpy as np
        f0 = np.array([200.0, 300.0, 400.0], dtype=np.float32)
        coarse, pitchf, _, _ = postprocess_f0(f0, device="cpu")
        self.assertEqual(pitchf.shape, (3,))
        self.assertAlmostEqual(pitchf[0].item(), 200.0, delta=1.0)

    def test_output_shapes(self):
        self._identity_config()
        f0 = torch.tensor([200.0, 300.0, 400.0])
        coarse, pitchf, _, _ = postprocess_f0(f0, device="cpu")
        self.assertEqual(coarse.shape, (3,))
        self.assertEqual(pitchf.shape, (3,))

    def test_coarse_dtype_long(self):
        self._identity_config()
        f0 = torch.tensor([200.0, 300.0])
        coarse, _, _, _ = postprocess_f0(f0, device="cpu")
        self.assertEqual(coarse.dtype, torch.long)

    def test_pitchf_dtype_float(self):
        self._identity_config()
        f0 = torch.tensor([200.0, 300.0])
        _, pitchf, _, _ = postprocess_f0(f0, device="cpu")
        self.assertEqual(pitchf.dtype, torch.float32)


class TestF0Constants(unittest.TestCase):
    """Tests for F0 related constants."""

    def test_fcpe_confidence_threshold(self):
        self.assertEqual(FCPE_CONFIDENCE_THRESHOLD, 0.025)








class TestCreateF0Extractor(unittest.TestCase):
    """Tests for create_f0_extractor factory."""

    def test_unknown_method_raises(self):
        """Unknown method raises F0ExtractionError."""
        with self.assertRaises(F0ExtractionError):
            create_f0_extractor("unknown", device=torch.device("cpu"), is_half=False, inference_cache=None)

    def test_error_message_contains_method(self):
        try:
            create_f0_extractor("bad_method", device=torch.device("cpu"), is_half=False, inference_cache=None)
        except F0ExtractionError as e:
            self.assertIn("bad_method", str(e))

    def test_f0extractor_is_abstract(self):
        """F0Extractor is abstract base class, cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            F0Extractor()


if __name__ == "__main__":
    unittest.main()
