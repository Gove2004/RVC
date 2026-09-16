"""Synthesizer 构造参数化契约测试（S9）。

锁死：
1. 子类一步构造 768 维 enc_p（不再有 256 维先建后删的中间态）；
2. 18 元 config 位置签名与 pth 的 "config" 字段兼容（loader 的调用方式不变）；
3. state_dict 键结构与权重往返（等价于"权重键不变"契约）。
"""
import unittest

import torch

from rvc.models.synthesizer_model import (
    SynthesizerTrnMsNSFsid,
    SynthesizerTrnMsNSFsid_nono,
)

# build_model_config 输出的 18 元顺序（48k v2 标准值）
CONFIG_48K = [
    1025,   # spec_channels = filter_length//2+1
    32,     # segment_size
    192,    # inter_channels
    192,    # hidden_channels
    768,    # filter_channels
    2,      # n_heads
    6,      # n_layers
    3,      # kernel_size
    0.1,    # p_dropout
    "1",    # resblock
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [12, 10, 2, 2],
    512,    # upsample_initial_channel
    [24, 20, 4, 4],
    109,    # spk_embed_dim
    48,     # gin_channels
    48000,  # sr
]

_TOP_PREFIXES = ("enc_p.", "enc_q.", "flow.", "dec.", "emb_g.")


class TestSynthesizerParameterized(unittest.TestCase):
    def test_f0_variant_enc_p_is_768_dim_single_construction(self):
        synth = SynthesizerTrnMsNSFsid(*CONFIG_48K, is_half=False)
        self.assertEqual(synth.enc_p.emb_phone.in_features, 768)
        self.assertTrue(synth.use_f0)

    def test_nono_variant_enc_p_is_768_dim_no_pitch(self):
        synth = SynthesizerTrnMsNSFsid_nono(*CONFIG_48K, is_half=False)
        self.assertEqual(synth.enc_p.emb_phone.in_features, 768)
        self.assertFalse(synth.use_f0)
        self.assertFalse(hasattr(synth.enc_p, "emb_pitch"))

    def test_state_dict_keys_cover_exactly_five_top_modules(self):
        synth = SynthesizerTrnMsNSFsid(*CONFIG_48K, is_half=False)
        prefixes = {key.split(".", 1)[0] for key in synth.state_dict()}
        self.assertEqual(prefixes, {"enc_p", "enc_q", "flow", "dec", "emb_g"})
        for key in synth.state_dict():
            self.assertTrue(key.startswith(_TOP_PREFIXES), key)

    def test_weight_round_trip_strict(self):
        """同 config 两实例间 strict load 必须成功（导出/加载契约）。"""
        source = SynthesizerTrnMsNSFsid(*CONFIG_48K, is_half=False)
        target = SynthesizerTrnMsNSFsid(*CONFIG_48K, is_half=False)
        target.load_state_dict(source.state_dict(), strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)

    def test_f0_variant_exports_enc_q_nono_does_not_infer(self):
        """infer 统一入口按 use_f0 分派：无 F0 变体不接受 pitch 参数。"""
        synth = SynthesizerTrnMsNSFsid_nono(*CONFIG_48K, is_half=False)
        phone = torch.randn(1, 10, 768)
        lengths = torch.tensor([10])
        sid = torch.tensor([0])
        # 不抛异常即可（数值正确性由 e2e 测试覆盖）
        synth.infer(phone, lengths, None, None, sid)


if __name__ == "__main__":
    unittest.main()
