"""HuBERT 特征器变体的缺省值契约。

项目在用的 .pth 全部派生自官方 v2 预训练（assets/pretrained/f0G40k.pth），
而那套预训练用的是 base 特征器。曾经缺省成 chinese，导致 enc_p 收到分布外
特征——表现是音色发闷、咬字怪，而且整条链路不报任何错（两套特征器同为
768 维，形状校验查不出来）。

这里锁死「不指定变体时必须落到 base」，任何一处缺省值被改回 chinese 都要
在本文件失败。chinese 仍是合法可选变体，只是不再是缺省。
"""
import unittest

from rvc.core.config import HUBERT_DEFAULT, HUBERT_VARIANTS, InferenceParams, OfflineParams
from rvc.models.hubert import hubert_path


class TestHubertDefaultVariant(unittest.TestCase):
    def test_default_is_base(self):
        self.assertEqual(HUBERT_DEFAULT, "base")

    def test_default_is_a_valid_variant(self):
        self.assertIn(HUBERT_DEFAULT, HUBERT_VARIANTS)

    def test_hubert_path_default_points_to_base(self):
        self.assertEqual(hubert_path(), hubert_path("base"))
        self.assertTrue(hubert_path().replace("\\", "/").endswith("/hubert/base"))

    def test_inference_params_default_is_base(self):
        self.assertEqual(InferenceParams().hubert, HUBERT_DEFAULT)

    def test_offline_params_default_is_base(self):
        self.assertEqual(OfflineParams.from_inference(
            InferenceParams(), input_path="a.wav", output_path="b.wav").hubert,
            HUBERT_DEFAULT)

    def test_signature_defaults_are_base(self):
        """函数签名缺省值也必须跟着走，否则绕过 config 的调用点会静默回到 chinese。"""
        import inspect

        from rvc.pipeline.pipeline import InferencePipeline
        from rvc.pipeline.sessions import ModelSessions
        from rvc.models.hubert import load_hubert
        from rvc.streaming.engine import VoiceEngine

        self.assertEqual(inspect.signature(load_hubert).parameters["variant"].default,
                         HUBERT_DEFAULT)
        self.assertEqual(inspect.signature(ModelSessions.load).parameters["hubert_variant"].default,
                         HUBERT_DEFAULT)
        self.assertEqual(inspect.signature(VoiceEngine.load_model).parameters["hubert"].default,
                         HUBERT_DEFAULT)
        self.assertEqual(inspect.signature(InferencePipeline.__init__).parameters["hubert"].default,
                         HUBERT_DEFAULT)

    def test_chinese_still_selectable(self):
        self.assertIn("chinese", HUBERT_VARIANTS)
        self.assertTrue(hubert_path("chinese").replace("\\", "/").endswith("/hubert/chinese"))


if __name__ == "__main__":
    unittest.main()
