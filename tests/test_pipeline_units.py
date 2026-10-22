"""管线单元契约 — 三处「静默算错」的回归锁定。

- engine.load_model 快路径必须同时比对 pth 与 hubert 变体（否则离线页选了 base
  却沿用 chinese 特征器，音色发闷且无告警）
- formant 重采样核缓存的键必须含两侧频率（upp_res 可在不同 target_sr 下撞出同值）
- normalize_f0_to_coarse 的 torch 分支必须与 numpy 分支逐值等价，且不使用
  布尔掩码索引（数据依赖形状 → 每块强制同步 + 无法进 CUDA Graph）
"""
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from rvc.pipeline.pitch.postprocess import normalize_f0_to_coarse
from rvc.pipeline.synthesis import apply_formant_resample, formant_resample_ratio
from rvc.streaming.engine import VoiceEngine


def _bare_engine(pipeline, pth):
    """绕过 __init__（它会构造 RuntimeDeviceConfig，需要真 CUDA）。"""
    engine = VoiceEngine.__new__(VoiceEngine)
    engine.pipeline = pipeline
    engine.pth_path = pth
    engine.inference_cache = None
    engine._cfg = mock.Mock()
    engine._load_lock = threading.Lock()
    return engine


class TestLoadModelFastPath(unittest.TestCase):
    def test_same_pth_and_variant_hits_fast_path(self):
        pipeline = mock.Mock(target_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor:
            self.assertEqual(engine.load_model("m.pth", hubert="chinese"), 40000)
            ctor.assert_not_called()

    def test_same_pth_different_variant_reloads(self):
        """实时页用 chinese 加载后，离线页切 base 必须真的重新加载。"""
        pipeline = mock.Mock(target_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor, \
                mock.patch("torch.cuda.empty_cache"), mock.patch("gc.collect"):
            ctor.return_value.target_sr = 40000
            engine.load_model("m.pth", hubert="base")
            ctor.assert_called_once()
            self.assertEqual(ctor.call_args.kwargs["hubert"], "base")

    def test_force_reloads_same_pth(self):
        pipeline = mock.Mock(target_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor, \
                mock.patch("torch.cuda.empty_cache"), mock.patch("gc.collect"):
            ctor.return_value.target_sr = 40000
            engine.load_model("m.pth", force=True, hubert="chinese")
            ctor.assert_called_once()


class TestFormantResampleKernelCache(unittest.TestCase):
    def test_ratio_is_derived_from_both_factor_and_sr(self):
        self.assertEqual(formant_resample_ratio(1.0, 40000), 400)
        self.assertEqual(formant_resample_ratio(2 ** (2 / 12), 40000), 448)

    def test_same_upp_res_across_sample_rates_uses_distinct_kernels(self):
        """490 这个 upp_res 在 48000 与 44100 下都能撞出来，核却不同。"""
        kernels = {}
        audio48 = torch.zeros(1, 4000)
        audio44 = torch.zeros(1, 4000)
        f48 = 490 / 480          # int(factor*48000//100) == 490
        f44 = 490 / 441          # int(factor*44100//100) == 490
        apply_formant_resample(audio48, f48, 48000, kernels, "cpu")
        apply_formant_resample(audio44, f44, 44100, kernels, "cpu")

        self.assertEqual(set(kernels), {(490, 480), (490, 441)},
                         "不同工作采样率必须各自持有重采样核")
        self.assertIsNot(kernels[(490, 480)], kernels[(490, 441)])

    def test_identity_when_ratio_matches(self):
        kernels = {}
        audio = torch.arange(8, dtype=torch.float32)
        out = apply_formant_resample(audio, 1.0, 40000, kernels, "cpu")
        self.assertIs(out, audio)
        self.assertEqual(kernels, {})


class TestNormalizeF0ToCoarse(unittest.TestCase):
    def _f0(self):
        rng = np.random.default_rng(7)
        return np.concatenate([
            np.zeros(16, dtype=np.float32),
            rng.uniform(40, 900, 200).astype(np.float32),
            np.array([0.0, np.nan, np.inf, -5.0], dtype=np.float32),
        ])

    def test_torch_matches_numpy_branch(self):
        f0 = self._f0()
        expected = normalize_f0_to_coarse(f0)
        got = normalize_f0_to_coarse(torch.from_numpy(f0)).numpy()
        self.assertTrue(np.array_equal(expected, got))

    def test_unvoiced_maps_to_one_and_range_clamped(self):
        out = normalize_f0_to_coarse(torch.from_numpy(self._f0())).numpy()
        self.assertEqual(out[:3].tolist(), [1, 1, 1])
        self.assertGreaterEqual(int(out.min()), 1)
        self.assertLessEqual(int(out.max()), 255)

    def test_no_boolean_mask_indexing(self):
        """布尔掩码索引会引入数据依赖形状，这里是「不许写」的硬约束。"""
        import inspect

        from rvc.pipeline.pitch import postprocess

        src = inspect.getsource(postprocess.normalize_f0_to_coarse)
        # 去掉注释行再检查（注释里为了解释禁令会提到 [voiced] 的写法）
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        torch_branch = code.split("else:", 1)[0]
        self.assertNotIn("[voiced]", torch_branch.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
