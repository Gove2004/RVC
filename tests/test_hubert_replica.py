"""HuBERT 纯 torch 复刻的无 GPU 契约测试。

复刻（rvc.models.hubert_replica）替代 transformers 作冻结特征器，必须守住三条
契约，任何一条破了都会让推理/训练静默出错：
  1. 输出帧数与 hubert_frame_count 的卷积折算完全一致（差一帧会让预生成 z 噪声场
     形状对不上 m_p，被形状校验静默拒绝）；
  2. eval 下前向确定（无 dropout/spec_augment/layerdrop 训练态分支）；
  3. weight_norm 新旧键名都能直装（base/ktum 用 original0/1，chinese 用 weight_g/v）。
全部用极小 config 在 CPU/fp32 上跑，不依赖真实权重与 GPU。
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rvc.models.hubert_replica import HubertReplica, load_replica
from rvc.pipeline.features import hubert_forward, hubert_frame_count


def _tiny_cfg():
    return SimpleNamespace(
        conv_dim=[8, 8],
        conv_kernel=[3, 3],
        conv_stride=[2, 2],
        conv_bias=False,
        num_feat_extract_layers=2,
        layer_norm_eps=1e-5,
        hidden_size=16,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        num_attention_heads=2,
        intermediate_size=32,
        num_hidden_layers=2,
        classifier_proj_size=8,
    )


class TestHubertReplica(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = HubertReplica(_tiny_cfg()).eval()

    def test_frame_count_matches_conv_fold(self):
        n = 1600
        wav = torch.randn(1, n)
        with torch.inference_mode():
            out = hubert_forward(self.model, wav)
        self.assertEqual(out.shape, (1, hubert_frame_count(self.model, n), 16))
        self.assertEqual(out.shape[1], hubert_frame_count(self.model, n))

    def test_eval_forward_is_deterministic(self):
        wav = torch.randn(1, 1600)
        with torch.inference_mode():
            a = hubert_forward(self.model, wav)
            b = hubert_forward(self.model, wav)
        self.assertTrue(torch.equal(a, b))

    def test_weight_norm_legacy_keys_load(self):
        """chinese 式旧键名 weight_g/weight_v 经兼容层映射到 original0/1，应全装。"""
        state = self.model.state_dict()
        renamed = {}
        for k, v in state.items():
            if k.endswith("parametrizations.weight.original0"):
                renamed[k.replace("parametrizations.weight.original0", "weight_g")] = v
            elif k.endswith("parametrizations.weight.original1"):
                renamed[k.replace("parametrizations.weight.original1", "weight_v")] = v
            else:
                renamed[k] = v
        fresh = HubertReplica(_tiny_cfg())
        result = fresh.load_state_dict(renamed, strict=False)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])

    def test_load_replica_roundtrip_bit_exact(self):
        """写 config.json + pytorch_model.bin 再 load_replica，前向应逐位一致。"""
        wav = torch.randn(1, 1600)
        with torch.inference_mode():
            ref = hubert_forward(self.model, wav)
        with tempfile.TemporaryDirectory() as d:
            cfg = _tiny_cfg()
            Path(d, "config.json").write_text(json.dumps(vars(cfg)), encoding="utf-8")
            torch.save(self.model.state_dict(), Path(d, "pytorch_model.bin"))
            loaded = load_replica(d, "cpu", is_half=False)
            self.assertIsInstance(loaded, HubertReplica)
            self.assertFalse(loaded.training)
            with torch.inference_mode():
                got = hubert_forward(loaded, wav)
            self.assertTrue(torch.equal(ref, got))


if __name__ == "__main__":
    unittest.main()