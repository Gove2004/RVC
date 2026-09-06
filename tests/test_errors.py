"""错误类型体系单元测试（unittest 风格）。"""
import unittest

from rvc.errors import (
    AudioDeviceError,
    ConfigError,
    FeatureExtractError,
    InferenceError,
    ModelLoadError,
    RVCError,
    format_error_message,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_all_errors_inherit_rvc_error(self):
        self.assertTrue(issubclass(ModelLoadError, RVCError))
        self.assertTrue(issubclass(AudioDeviceError, RVCError))
        self.assertTrue(issubclass(InferenceError, RVCError))
        self.assertTrue(issubclass(ConfigError, RVCError))
        self.assertTrue(issubclass(FeatureExtractError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(RVCError):
            raise ModelLoadError("test")

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


class TestFormatErrorMessage(unittest.TestCase):
    def test_rvc_error_passthrough(self):
        e = ModelLoadError("自定义消息")
        self.assertEqual(format_error_message(e), "自定义消息")

    def test_cuda_oom(self):
        e = RuntimeError("CUDA out of memory. Tried to allocate 100MB")
        msg = format_error_message(e)
        self.assertIn("显存不足", msg)

    def test_device_unavailable(self):
        e = RuntimeError("No device found")
        msg = format_error_message(e)
        self.assertIn("音频设备", msg)

    def test_generic_error(self):
        e = ValueError("some random error")
        msg = format_error_message(e)
        self.assertEqual(msg, "some random error")


if __name__ == "__main__":
    unittest.main()
