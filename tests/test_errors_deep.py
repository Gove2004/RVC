"""错误体系深度测试 — 所有异常类的继承、消息、捕获、链式。

覆盖：
- RVCError 基类
- ModelLoadError / AudioDeviceError / InferenceError
- F0ExtractionError / FeatureExtractionError（继承 InferenceError）
- ConfigError / AudioLoadError
- 异常捕获、链式异常、消息格式化
"""
import unittest

from rvc.core.errors import (
    AudioDeviceError,
    AudioLoadError,
    ConfigError,
    F0ExtractionError,
    FeatureExtractionError,
    InferenceError,
    ModelLoadError,
    RVCError,
)


class TestRVCErrorBase(unittest.TestCase):
    """RVCError 基类测试。"""

    def test_is_exception(self):
        self.assertTrue(issubclass(RVCError, Exception))

    def test_raise_and_catch(self):
        with self.assertRaises(RVCError):
            raise RVCError("test")

    def test_message(self):
        err = RVCError("hello world")
        self.assertEqual(str(err), "hello world")

    def test_empty_message(self):
        err = RVCError()
        self.assertEqual(str(err), "")

    def test_args(self):
        err = RVCError("a", "b", "c")
        self.assertEqual(err.args, ("a", "b", "c"))

    def test_catch_as_exception(self):
        with self.assertRaises(Exception):
            raise RVCError("test")

    def test_not_caught_as_value_error(self):
        with self.assertRaises(RVCError):
            try:
                raise RVCError("test")
            except ValueError:
                self.fail("RVCError 不应被 ValueError 捕获")

    def test_with_traceback(self):
        try:
            raise RVCError("with traceback")
        except RVCError as e:
            self.assertIsNotNone(e.__traceback__)

    def test_chained_exception(self):
        try:
            try:
                raise ValueError("original")
            except ValueError as e:
                raise RVCError("wrapped") from e
        except RVCError as e:
            self.assertIsInstance(e.__cause__, ValueError)
            self.assertEqual(str(e.__cause__), "original")

    def test_chained_implicit(self):
        try:
            try:
                raise ValueError("original")
            except ValueError:
                raise RVCError("wrapped")
        except RVCError as e:
            self.assertIsNotNone(e.__context__)
            self.assertIsInstance(e.__context__, ValueError)

    def test_custom_attributes(self):
        err = RVCError("test")
        err.custom = "value"
        self.assertEqual(err.custom, "value")

    def test_repr(self):
        err = RVCError("test")
        self.assertIn("RVCError", repr(err))
        self.assertIn("test", repr(err))


class TestModelLoadError(unittest.TestCase):
    """ModelLoadError 模型加载错误测试。"""

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(ModelLoadError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(ModelLoadError):
            raise ModelLoadError("model not found")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise ModelLoadError("test")

    def test_message(self):
        err = ModelLoadError("pth 文件损坏: /path/model.pth")
        self.assertIn("pth", str(err))
        self.assertIn("model.pth", str(err))

    def test_not_inference_error(self):
        self.assertFalse(issubclass(ModelLoadError, InferenceError))

    def test_with_path_context(self):
        path = "/models/test.pth"
        err = ModelLoadError(f"加载失败: {path}")
        self.assertIn(path, str(err))

    def test_chained_with_os_error(self):
        try:
            try:
                raise FileNotFoundError("file not found")
            except FileNotFoundError as e:
                raise ModelLoadError("模型文件不存在") from e
        except ModelLoadError as e:
            self.assertIsInstance(e.__cause__, FileNotFoundError)


class TestAudioDeviceError(unittest.TestCase):
    """AudioDeviceError 音频设备错误测试。"""

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(AudioDeviceError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(AudioDeviceError):
            raise AudioDeviceError("device not found")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise AudioDeviceError("test")

    def test_message(self):
        err = AudioDeviceError("设备索引 5 不存在")
        self.assertIn("5", str(err))

    def test_with_device_name(self):
        err = AudioDeviceError("设备「麦克风 (USB)」不支持立体声")
        self.assertIn("麦克风", str(err))
        self.assertIn("USB", str(err))

    def test_not_inference_error(self):
        self.assertFalse(issubclass(AudioDeviceError, InferenceError))

    def test_channel_count_message(self):
        err = AudioDeviceError("副输出设备只支持 1 通道，但主输出使用 2 通道")
        self.assertIn("1", str(err))
        self.assertIn("2", str(err))

    def test_chained_with_sounddevice_error(self):
        try:
            try:
                raise OSError("PortAudio error")
            except OSError as e:
                raise AudioDeviceError("打开设备失败") from e
        except AudioDeviceError as e:
            self.assertIsInstance(e.__cause__, OSError)


class TestInferenceError(unittest.TestCase):
    """InferenceError 推理错误测试。"""

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(InferenceError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(InferenceError):
            raise InferenceError("inference failed")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise InferenceError("test")

    def test_message(self):
        err = InferenceError("合成器推理失败: CUDA out of memory")
        self.assertIn("CUDA", str(err))

    def test_not_model_load_error(self):
        self.assertFalse(issubclass(InferenceError, ModelLoadError))

    def test_with_block_info(self):
        err = InferenceError("第 15 块推理失败")
        self.assertIn("15", str(err))

    def test_chained_with_runtime_error(self):
        try:
            try:
                raise RuntimeError("CUDA error")
            except RuntimeError as e:
                raise InferenceError("推理运行时错误") from e
        except InferenceError as e:
            self.assertIsInstance(e.__cause__, RuntimeError)


class TestF0ExtractionError(unittest.TestCase):
    """F0ExtractionError F0 提取错误测试。"""

    def test_inherits_inference_error(self):
        self.assertTrue(issubclass(F0ExtractionError, InferenceError))

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(F0ExtractionError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(F0ExtractionError):
            raise F0ExtractionError("RMVPE 权重缺失")

    def test_catch_as_inference_error(self):
        with self.assertRaises(InferenceError):
            raise F0ExtractionError("test")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise F0ExtractionError("test")

    def test_message(self):
        err = F0ExtractionError("RMVPE 模型推理异常: 输入长度不足")
        self.assertIn("RMVPE", str(err))

    def test_not_feature_extraction_error(self):
        self.assertFalse(issubclass(F0ExtractionError, FeatureExtractionError))

    def test_with_method_info(self):
        for method in ["rmvpe", "fcpe", "crepe", "pm", "harvest"]:
            err = F0ExtractionError(f"{method} 提取失败")
            self.assertIn(method, str(err))


class TestFeatureExtractionError(unittest.TestCase):
    """FeatureExtractionError 特征提取错误测试。"""

    def test_inherits_inference_error(self):
        self.assertTrue(issubclass(FeatureExtractionError, InferenceError))

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(FeatureExtractionError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(FeatureExtractionError):
            raise FeatureExtractionError("HuBERT 特征提取失败")

    def test_catch_as_inference_error(self):
        with self.assertRaises(InferenceError):
            raise FeatureExtractionError("test")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise FeatureExtractionError("test")

    def test_message(self):
        err = FeatureExtractionError("HuBERT 模型推理异常")
        self.assertIn("HuBERT", str(err))

    def test_not_f0_extraction_error(self):
        self.assertFalse(issubclass(FeatureExtractionError, F0ExtractionError))

    def test_with_variant_info(self):
        for variant in ["chinese", "base", "japanese"]:
            err = FeatureExtractionError(f"HuBERT ({variant}) 提取失败")
            self.assertIn(variant, str(err))


class TestConfigError(unittest.TestCase):
    """ConfigError 配置错误测试。"""

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(ConfigError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(ConfigError):
            raise ConfigError("配置校验失败")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise ConfigError("test")

    def test_message(self):
        err = ConfigError("pitch 必须在 [-12, 12] 范围内，当前值: 20")
        self.assertIn("pitch", str(err))
        self.assertIn("20", str(err))

    def test_not_inference_error(self):
        self.assertFalse(issubclass(ConfigError, InferenceError))

    def test_with_field_name(self):
        for field in ["pitch", "formant", "protect", "rms_mix", "block_time"]:
            err = ConfigError(f"{field} 参数越界")
            self.assertIn(field, str(err))

    def test_chained_with_value_error(self):
        try:
            try:
                raise ValueError("invalid value")
            except ValueError as e:
                raise ConfigError("配置解析失败") from e
        except ConfigError as e:
            self.assertIsInstance(e.__cause__, ValueError)


class TestAudioLoadError(unittest.TestCase):
    """AudioLoadError 音频加载错误测试。"""

    def test_inherits_rvc_error(self):
        self.assertTrue(issubclass(AudioLoadError, RVCError))

    def test_raise_and_catch(self):
        with self.assertRaises(AudioLoadError):
            raise AudioLoadError("音频加载失败")

    def test_catch_as_rvc_error(self):
        with self.assertRaises(RVCError):
            raise AudioLoadError("test")

    def test_message(self):
        err = AudioLoadError("文件不存在: /path/audio.wav")
        self.assertIn("audio.wav", str(err))

    def test_not_inference_error(self):
        self.assertFalse(issubclass(AudioLoadError, InferenceError))

    def test_with_format_info(self):
        for fmt in ["wav", "mp3", "flac", "ogg", "m4a"]:
            err = AudioLoadError(f"不支持的格式: {fmt}")
            self.assertIn(fmt, str(err))

    def test_chained_with_file_not_found(self):
        try:
            try:
                raise FileNotFoundError("no such file")
            except FileNotFoundError as e:
                raise AudioLoadError("音频文件不存在") from e
        except AudioLoadError as e:
            self.assertIsInstance(e.__cause__, FileNotFoundError)


class TestExceptionHierarchy(unittest.TestCase):
    """异常继承体系整体测试。"""

    def test_all_inherit_rvc_error(self):
        for cls in [
            ModelLoadError, AudioDeviceError, InferenceError,
            F0ExtractionError, FeatureExtractionError, ConfigError, AudioLoadError,
        ]:
            self.assertTrue(issubclass(cls, RVCError), f"{cls.__name__} 应继承 RVCError")

    def test_f0_and_feature_inherit_inference(self):
        self.assertTrue(issubclass(F0ExtractionError, InferenceError))
        self.assertTrue(issubclass(FeatureExtractionError, InferenceError))

    def test_catch_all_as_rvc_error(self):
        """所有自定义异常都能被 RVCError 捕获。"""
        for cls in [
            ModelLoadError, AudioDeviceError, InferenceError,
            F0ExtractionError, FeatureExtractionError, ConfigError, AudioLoadError,
        ]:
            with self.assertRaises(RVCError):
                raise cls("test")

    def test_catch_inference_subtypes(self):
        """InferenceError 能捕获 F0 和 Feature 错误。"""
        with self.assertRaises(InferenceError):
            raise F0ExtractionError("test")
        with self.assertRaises(InferenceError):
            raise FeatureExtractionError("test")

    def test_distinct_types_not_cross_caught(self):
        """不同类型的异常不会互相捕获。"""
        with self.assertRaises(ModelLoadError):
            try:
                raise ModelLoadError("test")
            except AudioDeviceError:
                self.fail("ModelLoadError 不应被 AudioDeviceError 捕获")

    def test_exception_count(self):
        """共有 8 个自定义异常类（1 基类 + 7 子类）。"""
        classes = [
            RVCError, ModelLoadError, AudioDeviceError, InferenceError,
            F0ExtractionError, FeatureExtractionError, ConfigError, AudioLoadError,
        ]
        self.assertEqual(len(classes), 8)


if __name__ == "__main__":
    unittest.main()
