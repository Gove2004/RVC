"""错误类型单元测试（unittest 风格，无需额外依赖）。"""
import unittest

from rvc.errors import ModelLoadError


class TestModelLoadError(unittest.TestCase):
    def test_error_message(self):
        e = ModelLoadError("模型加载失败")
        self.assertEqual(str(e), "模型加载失败")

    def test_exception_chain(self):
        try:
            try:
                raise ValueError("原始错误")
            except ValueError as orig:
                raise ModelLoadError("包装错误") from orig
        except ModelLoadError as e:
            self.assertIsInstance(e.__cause__, ValueError)
            self.assertEqual(str(e.__cause__), "原始错误")


if __name__ == "__main__":
    unittest.main()
