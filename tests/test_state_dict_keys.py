"""权重键名归一化契约 — weight_norm 两套命名必须能互通。

背景：上游导出的模型与官方预训练资产用旧版 weight_norm 命名
（``weight_g``/``weight_v``），本项目模块用 ``torch.nn.utils.parametrize``
（``parametrizations.weight.original0/1``）。``load_state_dict(strict=False)``
对键名不匹配是静默的，不做归一化就整层跳过：输出照样能听，但解码器上采样
等参数停在随机初始化。这里锁的是「键名必须能对上」这个最低契约。
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rvc.models.state_dict_keys import to_legacy, to_parametrized, weight_norm_style


def _legacy_state():
    return {
        "enc_p.emb_phone.weight": torch.arange(4, dtype=torch.float32),
        "dec.ups.0.weight_g": torch.ones(3, 1, 1),
        "dec.ups.0.weight_v": torch.full((3, 2, 2), 2.0),
        "dec.ups.0.bias": torch.zeros(3),
    }


class TestToParametrized(unittest.TestCase):
    def test_rewrites_legacy_keys(self):
        state = to_parametrized(_legacy_state())

        self.assertIn("dec.ups.0.parametrizations.weight.original0", state)
        self.assertIn("dec.ups.0.parametrizations.weight.original1", state)
        self.assertNotIn("dec.ups.0.weight_g", state)
        self.assertNotIn("dec.ups.0.weight_v", state)
        # 非 weight_norm 的键原样保留
        self.assertIn("enc_p.emb_phone.weight", state)
        self.assertIn("dec.ups.0.bias", state)
        self.assertEqual(len(state), 4)

    def test_preserves_values(self):
        legacy = _legacy_state()
        state = to_parametrized(legacy)

        self.assertTrue(torch.equal(
            state["dec.ups.0.parametrizations.weight.original0"], legacy["dec.ups.0.weight_g"]))
        self.assertTrue(torch.equal(
            state["dec.ups.0.parametrizations.weight.original1"], legacy["dec.ups.0.weight_v"]))

    def test_noop_returns_same_object(self):
        """已是 parametrize 命名时不做拷贝（457 张量的字典不必白复制一份）。"""
        state = to_parametrized(_legacy_state())

        self.assertIs(to_parametrized(state), state)


class TestToLegacy(unittest.TestCase):
    def test_rewrites_parametrized_keys(self):
        state = to_legacy(to_parametrized(_legacy_state()))

        self.assertIn("dec.ups.0.weight_g", state)
        self.assertIn("dec.ups.0.weight_v", state)
        self.assertNotIn("dec.ups.0.parametrizations.weight.original0", state)

    def test_round_trip(self):
        legacy = _legacy_state()
        back = to_legacy(to_parametrized(legacy))

        self.assertEqual(set(back), set(legacy))
        for key, value in legacy.items():
            self.assertTrue(torch.equal(back[key], value), key)


class TestWeightNormStyle(unittest.TestCase):
    def test_detects_each_style(self):
        self.assertEqual(weight_norm_style(_legacy_state()), "legacy")
        self.assertEqual(weight_norm_style(to_parametrized(_legacy_state())), "parametrized")
        self.assertEqual(weight_norm_style({"a.weight": torch.zeros(1)}), "none")
        self.assertEqual(
            weight_norm_style({**_legacy_state(), **to_parametrized(_legacy_state())}), "mixed")


class TestInferenceLoaderLoadsLegacyState(unittest.TestCase):
    """推理侧加载旧版命名模型时，weight_norm 层必须全部落地。"""

    def test_all_weight_norm_tensors_land(self):
        from rvc.models.synthesizer import SynthesizerTrnMsNSFsid
        from rvc.pipeline.loader import SynthesizerLoader

        config = [513, 32, 192, 192, 768, 2, 6, 3, 0.0, "1", [3, 7, 11],
                  [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]
        source = SynthesizerTrnMsNSFsid(*config, is_half=False)
        del source.enc_q
        saved = to_legacy(source.state_dict())

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pth")
            torch.save({"weight": saved, "config": config, "info": "t",
                        "sr": "40k", "f0": 1, "version": "v2"}, path)
            bundle = SynthesizerLoader(SimpleNamespace(device="cpu", is_half=False)).load(path)

        now = bundle.synthesizer.state_dict()
        normalized = to_parametrized(saved)
        not_loaded = [k for k in now if k not in normalized]
        self.assertEqual(not_loaded, [], "weight_norm 层未落地，解码器会停在随机初始化")
        for key in now:
            self.assertTrue(torch.equal(normalized[key], now[key]), key)


class TestExportWritesLegacyState(unittest.TestCase):
    """导出落盘用旧版命名，与本仓库已有模型文件、上游工具保持一致。"""

    def test_export_uses_legacy_keys_and_round_trips(self):
        from rvc.models.synthesizer import SynthesizerTrnMsNSFsid
        from rvc.pipeline.loader import SynthesizerLoader
        from rvc.train.checkpoint import export_model, load_train_json

        config = [513, 32, 192, 192, 768, 2, 6, 3, 0.0, "1", [3, 7, 11],
                  [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]
        source = SynthesizerTrnMsNSFsid(*config, is_half=False)
        del source.enc_q

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "exported.pth")
            export_model(source.state_dict(), 40000, load_train_json(40000), 1, path)
            exported = torch.load(path, map_location="cpu", weights_only=False)

            self.assertEqual(weight_norm_style(exported["weight"]), "legacy")
            bundle = SynthesizerLoader(SimpleNamespace(device="cpu", is_half=False)).load(path)

        now = bundle.synthesizer.state_dict()
        normalized = to_parametrized(exported["weight"])
        self.assertEqual([k for k in now if k not in normalized], [])


class TestMergeAcrossNamingStyles(unittest.TestCase):
    """跨命名合并时同一层必须被加权平均，而不是当成两个键各留一份。"""

    def test_cross_style_merge_averages_same_layer(self):
        from rvc.train.checkpoint import merge_models

        config = [513, 32, 192, 192, 768, 2, 6, 3, 0.0, "1", [3, 7, 11],
                  [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]
        legacy = {"dec.ups.0.weight_g": torch.ones(3, 1, 1),
                  "dec.ups.0.weight_v": torch.full((3, 2, 2), 2.0),
                  "dec.ups.0.bias": torch.zeros(3)}
        new_style = to_parametrized({"dec.ups.0.weight_g": torch.full((3, 1, 1), 3.0),
                                     "dec.ups.0.weight_v": torch.full((3, 2, 2), 4.0),
                                     "dec.ups.0.bias": torch.zeros(3)})
        meta = {"config": config, "info": "m", "sr": "40k", "f0": 1, "version": "v2"}

        with tempfile.TemporaryDirectory() as tmp:
            path_a = str(Path(tmp) / "a.pth")
            path_b = str(Path(tmp) / "b.pth")
            out = str(Path(tmp) / "merged.pth")
            torch.save({**meta, "weight": legacy}, path_a)
            torch.save({**meta, "weight": new_style}, path_b)
            merge_models(path_a, path_b, 0.5, out)
            merged = torch.load(out, map_location="cpu", weights_only=False)["weight"]

        self.assertEqual(len(merged), 3, "跨命名合并丢了层或留了重复层")
        merged = to_parametrized(merged)
        # 落盘统一转 fp16，比较前转回 float32
        self.assertTrue(torch.allclose(
            merged["dec.ups.0.parametrizations.weight.original0"].float(),
            torch.full((3, 1, 1), 2.0)))
        self.assertTrue(torch.allclose(
            merged["dec.ups.0.parametrizations.weight.original1"].float(),
            torch.full((3, 2, 2), 3.0)))


if __name__ == "__main__":
    unittest.main()
