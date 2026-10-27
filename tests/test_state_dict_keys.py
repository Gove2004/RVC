"""权重键名契约 — weight_norm 两套命名的边界。

背景：上游模型与官方预训练资产用旧版 weight_norm 键名
（``weight_g``/``weight_v``），本项目模块用 ``parametrizations.weight_norm``
（``parametrizations.weight.original0/1``）。

两条边界要分开看：

- **加载**：torch 的 ``ParametrizationList`` 自带旧键名兼容层，旧命名模型
  能直接正确加载，不需要手写改写。这里锁住这个前提——一旦哪天不成立，
  用户手上的模型会静默变成随机权重
- **两份字典互相比对**（``merge_models``）：没有兼容层，必须先归一化，
  否则同一层被当成两个键，权重不参与平均
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.utils import parametrizations

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


class TestLegacyKeysAcceptedOnLoad(unittest.TestCase):
    """加载侧不需要改写：torch 自带旧键名兼容层。

    这条是整个「旧命名模型能否用」的前提。若哪天 torch 去掉兼容层，
    用户手上的模型会静默变成随机权重（strict=False 不报错），
    所以必须显式锁住，而不是依赖 state_dict() 键名比对去推断。
    """

    def test_parametrization_list_accepts_weight_g_and_v(self):
        source = torch.nn.Conv1d(4, 6, 3)
        parametrizations.weight_norm(source, name="weight")
        with torch.no_grad():
            source.parametrizations.weight.original0.fill_(2.5)
            source.parametrizations.weight.original1.normal_()

        legacy = {
            "weight_g": source.parametrizations.weight.original0.detach().clone(),
            "weight_v": source.parametrizations.weight.original1.detach().clone(),
            "bias": source.bias.detach().clone(),
        }

        target = torch.nn.Conv1d(4, 6, 3)
        parametrizations.weight_norm(target, name="weight")
        result = target.load_state_dict(legacy, strict=False)

        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        self.assertTrue(torch.equal(
            target.parametrizations.weight.original0, legacy["weight_g"]))
        self.assertTrue(torch.equal(
            target.parametrizations.weight.original1, legacy["weight_v"]))
        self.assertTrue(torch.equal(target.weight, source.weight))


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
