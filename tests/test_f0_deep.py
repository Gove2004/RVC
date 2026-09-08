"""F0 extractor deep tests - pure functions only.

Covers:
- _FilteredStream: prefix/contains filtering, empty text, flush, __getattr__
- _normalize_f0_to_coarse: various F0 values, boundaries, UV
- apply_f0_break_protect: thresholds, compression ratio, knee, boundaries
- postprocess_f0: pitch shift, break protect toggle, numpy/tensor input
- constants: FCPE_CONFIDENCE_THRESHOLD / BREAK_PROTECT_*
- create_f0_extractor: unknown method raises
"""
import io
import sys
import unittest

import torch

from rvc.inference.f0_extractor import (
    BREAK_PROTECT_DEFAULT_KNEE,
    BREAK_PROTECT_DEFAULT_RATIO,
    BREAK_PROTECT_DEFAULT_SRC_HZ,
    FCPE_CONFIDENCE_THRESHOLD,
    F0Extractor,
    _FilteredStream,
    _normalize_f0_to_coarse,
    _suppress_third_party_output,
    apply_f0_break_protect,
    create_f0_extractor,
    postprocess_f0,
)
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
        coarse = _normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 1)

    def test_negative_f0_returns_one(self):
        """Negative F0 maps to 1."""
        f0 = torch.tensor([-10.0])
        coarse = _normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 1)

    def test_very_high_f0_clamped_to_255(self):
        """Very high F0 is clamped to 255."""
        f0 = torch.tensor([10000.0])
        coarse = _normalize_f0_to_coarse(f0)
        self.assertEqual(coarse[0].item(), 255)

    def test_normal_f0_in_range(self):
        """Normal F0 (100-1000 Hz) is in 1-255 range."""
        for hz in [100, 200, 300, 440, 500, 800, 1000]:
            f0 = torch.tensor([float(hz)])
            coarse = _normalize_f0_to_coarse(f0)
            self.assertGreaterEqual(coarse[0].item(), 1)
            self.assertLessEqual(coarse[0].item(), 255)

    def test_higher_f0_higher_coarse(self):
        """Higher F0 gives higher coarse value (monotonic)."""
        f0_low = torch.tensor([200.0])
        f0_high = torch.tensor([400.0])
        coarse_low = _normalize_f0_to_coarse(f0_low)
        coarse_high = _normalize_f0_to_coarse(f0_high)
        self.assertGreater(coarse_high[0].item(), coarse_low[0].item())

    def test_output_dtype_long(self):
        f0 = torch.tensor([440.0, 500.0])
        coarse = _normalize_f0_to_coarse(f0)
        self.assertEqual(coarse.dtype, torch.long)

    def test_batch_input(self):
        f0 = torch.tensor([0.0, 200.0, 440.0, 10000.0])
        coarse = _normalize_f0_to_coarse(f0)
        self.assertEqual(coarse.shape, (4,))
        self.assertEqual(coarse[0].item(), 1)
        self.assertEqual(coarse[3].item(), 255)

    def test_mid_f0_around_128(self):
        """Mid F0 should be around 128."""
        f0 = torch.tensor([500.0])
        coarse = _normalize_f0_to_coarse(f0)
        self.assertGreater(coarse[0].item(), 50)
        self.assertLess(coarse[0].item(), 200)


class TestApplyF0BreakProtect(unittest.TestCase):
    """Tests for apply_f0_break_protect."""

    def test_below_critical_unchanged(self):
        """F0 below critical stays unchanged."""
        f0 = torch.tensor([100.0, 200.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.0)
        self.assertTrue(torch.allclose(result, f0))

    def test_above_critical_compressed(self):
        """F0 above critical is compressed."""
        f0 = torch.tensor([500.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.0)
        # y = C + ratio * (x - C) = 300 + 0.4 * 200 = 380
        self.assertAlmostEqual(result[0].item(), 380.0, delta=1.0)

    def test_ratio_one_unchanged(self):
        """ratio=1 means no compression."""
        f0 = torch.tensor([500.0, 1000.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=1.0, knee=0.0)
        self.assertTrue(torch.allclose(result, f0))

    def test_critical_zero_returns_input(self):
        """critical_hz<=0 returns input unchanged."""
        f0 = torch.tensor([500.0])
        result = apply_f0_break_protect(f0, critical_hz=0.0, ratio=0.4, knee=0.0)
        self.assertTrue(torch.allclose(result, f0))

    def test_critical_negative_returns_input(self):
        f0 = torch.tensor([500.0])
        result = apply_f0_break_protect(f0, critical_hz=-100.0, ratio=0.4, knee=0.0)
        self.assertTrue(torch.allclose(result, f0))

    def test_none_input_returns_none(self):
        result = apply_f0_break_protect(None, critical_hz=300.0)
        self.assertIsNone(result)

    def test_knee_smooths_transition(self):
        """knee > 0 smooths the transition (value between hard knee and original)."""
        f0 = torch.tensor([300.0])  # exactly at boundary
        result_hard = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.0)
        result_soft = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.12)
        # knee smoothing should give value close to hard knee at boundary
        self.assertAlmostEqual(result_hard[0].item(), result_soft[0].item(), delta=10.0)

    def test_knee_zero_hard_knee(self):
        """knee=0 is hard knee."""
        f0 = torch.tensor([299.0, 301.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.5, knee=0.0)
        # 299 < 300 -> unchanged 299
        self.assertAlmostEqual(result[0].item(), 299.0, delta=0.1)
        # 301 > 300 -> 300 + 0.5 * 1 = 300.5
        self.assertAlmostEqual(result[1].item(), 300.5, delta=0.1)

    def test_monotonic(self):
        """Output is monotonically increasing."""
        f0 = torch.linspace(100, 1000, 100)
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.12)
        diffs = torch.diff(result)
        self.assertTrue(torch.all(diffs >= -0.01))  # allow tiny numerical error

    def test_various_ratios(self):
        for ratio in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            f0 = torch.tensor([600.0, 700.0])
            result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=ratio, knee=0.0)
            expected = 300 + ratio * 300
            self.assertAlmostEqual(result[0].item(), expected, delta=1.0)

    def test_various_criticals(self):
        for critical in [200, 300, 400, 500]:
            f0 = torch.tensor([100.0])  # below all criticals
            result = apply_f0_break_protect(f0, critical_hz=float(critical), ratio=0.4, knee=0.0)
            self.assertAlmostEqual(result[0].item(), 100.0, delta=0.1)

    def test_batch_input(self):
        f0 = torch.tensor([100.0, 300.0, 500.0, 1000.0])
        result = apply_f0_break_protect(f0, critical_hz=300.0, ratio=0.4, knee=0.0)
        self.assertEqual(result.shape, (4,))
        self.assertAlmostEqual(result[0].item(), 100.0, delta=0.1)
        self.assertAlmostEqual(result[2].item(), 380.0, delta=1.0)


class TestPostprocessF0(unittest.TestCase):
    """Tests for postprocess_f0."""

    def test_zero_shift_unchanged(self):
        """f0_up_key=0 means pitch unchanged."""
        f0 = torch.tensor([440.0, 500.0])
        coarse, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu")
        self.assertAlmostEqual(pitchf[0].item(), 440.0, delta=1.0)

    def test_positive_shift_increases(self):
        """Positive semitone shift increases pitch."""
        f0 = torch.tensor([440.0, 500.0])
        _, pitchf = postprocess_f0(f0, f0_up_key=12, device="cpu")
        # 12 semitones = 1 octave = 2x
        self.assertAlmostEqual(pitchf[0].item(), 880.0, delta=2.0)

    def test_negative_shift_decreases(self):
        """Negative semitone shift decreases pitch."""
        f0 = torch.tensor([440.0, 500.0])
        _, pitchf = postprocess_f0(f0, f0_up_key=-12, device="cpu")
        self.assertAlmostEqual(pitchf[0].item(), 220.0, delta=1.0)

    def test_numpy_input(self):
        """numpy array input also works."""
        import numpy as np
        f0 = np.array([440.0, 500.0, 600.0], dtype=np.float32)
        coarse, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu")
        self.assertEqual(pitchf.shape, (3,))
        self.assertAlmostEqual(pitchf[0].item(), 440.0, delta=1.0)

    def test_break_protect_enabled(self):
        """Break protect enabled compresses high pitch."""
        f0 = torch.tensor([600.0, 700.0])
        # f0_proc = (enabled, critical_hz, ratio, knee)
        f0_proc = (True, 300.0, 0.4, 0.0)
        _, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu", f0_proc=f0_proc)
        # critical 300, 600 > 300, compressed = 300 + 0.4 * 300 = 420
        self.assertAlmostEqual(pitchf[0].item(), 420.0, delta=2.0)

    def test_break_protect_disabled(self):
        """Break protect disabled does not compress."""
        f0 = torch.tensor([600.0, 700.0])
        f0_proc = (False, 300.0, 0.4, 0.0)
        _, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu", f0_proc=f0_proc)
        self.assertAlmostEqual(pitchf[0].item(), 600.0, delta=1.0)

    def test_break_protect_none(self):
        """f0_proc=None means no compression."""
        f0 = torch.tensor([600.0, 700.0])
        _, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu", f0_proc=None)
        self.assertAlmostEqual(pitchf[0].item(), 600.0, delta=1.0)

    def test_output_shapes(self):
        f0 = torch.tensor([440.0, 500.0, 600.0])
        coarse, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu")
        self.assertEqual(coarse.shape, (3,))
        self.assertEqual(pitchf.shape, (3,))

    def test_coarse_dtype_long(self):
        f0 = torch.tensor([440.0, 500.0])
        coarse, _ = postprocess_f0(f0, f0_up_key=0, device="cpu")
        self.assertEqual(coarse.dtype, torch.long)

    def test_pitchf_dtype_float(self):
        f0 = torch.tensor([440.0, 500.0])
        _, pitchf = postprocess_f0(f0, f0_up_key=0, device="cpu")
        self.assertEqual(pitchf.dtype, torch.float32)


class TestF0Constants(unittest.TestCase):
    """Tests for F0 related constants."""

    def test_fcpe_confidence_threshold(self):
        self.assertEqual(FCPE_CONFIDENCE_THRESHOLD, 0.025)

    def test_break_protect_default_src_hz(self):
        self.assertEqual(BREAK_PROTECT_DEFAULT_SRC_HZ, 300.0)

    def test_break_protect_default_ratio(self):
        self.assertEqual(BREAK_PROTECT_DEFAULT_RATIO, 0.4)

    def test_break_protect_default_knee(self):
        self.assertEqual(BREAK_PROTECT_DEFAULT_KNEE, 0.12)

    def test_ratio_between_0_and_1(self):
        self.assertGreater(BREAK_PROTECT_DEFAULT_RATIO, 0)
        self.assertLess(BREAK_PROTECT_DEFAULT_RATIO, 1)

    def test_knee_positive(self):
        self.assertGreater(BREAK_PROTECT_DEFAULT_KNEE, 0)


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
