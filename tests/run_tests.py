"""单元测试运行器 — unittest 自动发现，零第三方依赖。

新测试文件放进本目录即被自动发现，无需登记（旧手工清单已废弃）。

运行: python -m tests.run_tests
"""
import sys
import unittest
from pathlib import Path


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir), top_level_dir=str(project_root)
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
