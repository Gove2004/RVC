"""配置数据类测试 — 验证 update_from 原地复制语义。"""
import unittest

from rvc.core.config import InferenceParams, OfflineParams


class TestInferenceParams(unittest.TestCase):
    def test_default_values(self):
        p = InferenceParams()
        assert p.rms_mix == 0.0
        assert p.model_path == ""
        assert p.hubert in ("base", "chinese")

    def test_update_from_copies_fields(self):
        src = InferenceParams()
        src.rms_mix = 0.7
        src.model_path = "/tmp/model.pth"
        src.hubert = "chinese"
        src.voice.formant = 0.5
        src.f0.rmvpe_threshold = 0.03
        src.buffer.block_time = 4096

        dst = InferenceParams()
        dst.update_from(src)

        assert dst.rms_mix == 0.7
        assert dst.model_path == "/tmp/model.pth"
        assert dst.hubert == "chinese"
        assert dst.voice.formant == 0.5
        assert dst.f0.rmvpe_threshold == 0.03
        assert dst.buffer.block_time == 4096

    def test_update_from_preserves_reference(self):
        """update_from 不替换对象引用，只修改字段值。"""
        dst = InferenceParams()
        original_voice_ref = dst.voice
        src = InferenceParams()
        src.voice.formant = 0.8

        dst.update_from(src)

        # voice 分组对象引用不变
        assert dst.voice is original_voice_ref
        assert dst.voice.formant == 0.8

    def test_update_from_independent(self):
        """修改源对象不影响已 update_from 的目标对象。"""
        src = InferenceParams()
        src.rms_mix = 0.5
        dst = InferenceParams()
        dst.update_from(src)
        src.rms_mix = 0.9
        assert dst.rms_mix == 0.5


class TestOfflineParams(unittest.TestCase):
    def test_inherits_inference_fields(self):
        p = OfflineParams()
        assert hasattr(p, "rms_mix")
        assert hasattr(p, "voice")
        assert p.input_path == ""
        assert p.output_path == ""

    def test_update_from_offline_to_inference(self):
        """OfflineParams 可以 update_from 到 InferenceParams（只复制共有字段）。"""
        src = OfflineParams()
        src.rms_mix = 0.3
        src.input_path = "/tmp/in.wav"
        src.output_path = "/tmp/out.wav"

        dst = InferenceParams()
        dst.update_from(src)

        assert dst.rms_mix == 0.3
        # InferenceParams 没有 input_path/output_path 字段
        assert not hasattr(dst, "input_path")
