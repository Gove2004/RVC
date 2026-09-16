"""纯函数单元测试运行器 — 不依赖 pytest，用标准库 unittest。

运行: python -m tests.run_tests
"""
import sys
import unittest

from tests.test_config import TestInferenceParams, TestOfflineParams
from tests.test_pitch_postprocess import (
    TestApplyPitchMap,
    TestConstants,
    TestHzMidiConversion,
    TestMedianFilterF0,
    TestNormalizeF0ToCoarse,
)
from tests.test_rms_mix import TestApplyRmsMix, TestFastRms


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in (
        TestHzMidiConversion,
        TestApplyPitchMap,
        TestMedianFilterF0,
        TestNormalizeF0ToCoarse,
        TestConstants,
        TestFastRms,
        TestApplyRmsMix,
        TestInferenceParams,
        TestOfflineParams,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
