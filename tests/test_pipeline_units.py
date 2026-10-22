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


class _FakeSynth:
    def __init__(self):
        self.removed = 0

    def remove_weight_norm(self):
        self.removed += 1


class _FakeBundle:
    def __init__(self):
        self.synthesizer = _FakeSynth()
        self.target_sr = 40000
        self.use_f0 = True


class _FakeEntry:
    def __init__(self):
        self.lock = threading.RLock()


class _FakeGraphCache:
    """假图缓存（真 _GraphCache 需要 CUDA 才能捕获）。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.entries = {("ns",): _FakeEntry()}


def _patched_load():
    """加载器替换为假实现的 patcher 列表（真加载需要权重与 transformers）。"""
    return [
        mock.patch("rvc.pipeline.sessions.load_hubert"),
        mock.patch("rvc.pipeline.sessions.SynthesizerLoader"),
        mock.patch("rvc.pipeline.sessions.torch.cuda.is_available", return_value=False),
        mock.patch("rvc.pipeline.sessions.gc.collect"),
    ]


class TestModelSessionsCache(unittest.TestCase):
    """会话层缓存：命中要「零后处理 + 保留图缓存」，否则 LRU 等于白做。"""

    def setUp(self):
        from rvc.pipeline.cache import InferenceCache
        from rvc.pipeline.sessions import ModelSessions

        self.sessions = ModelSessions("cpu", True, InferenceCache())
        self._patchers = _patched_load()
        started = [p.start() for p in self._patchers]
        self.addCleanup(lambda: [p.stop() for p in self._patchers])
        self.hubert, self.loader = started[0], started[1]
        self.hubert.return_value = object()
        self.loader.return_value.load.return_value = _FakeBundle()

    def test_same_pth_loads_once_and_skips_post_processing(self):
        first = self.sessions.load("a.pth")
        second = self.sessions.load("a.pth")

        self.assertIs(first.synthesizer, second.synthesizer)
        self.assertEqual(self.loader.return_value.load.call_count, 1)
        self.assertEqual(self.hubert.call_count, 1)
        # reloaded=False → 不再重复 remove_weight_norm
        self.assertEqual(first.synthesizer.removed, 1)

    def test_cache_hit_keeps_captured_graphs(self):
        first = self.sessions.load("a.pth")
        sentinel = _FakeGraphCache()
        first.synthesizer._rvc_cuda_graph_cache = sentinel

        second = self.sessions.load("a.pth")

        self.assertIs(getattr(second.synthesizer, "_rvc_cuda_graph_cache", None),
                      sentinel,
                      "切回已缓存模型时图缓存必须存活，不能被无条件 purge")

    def test_hubert_variant_change_reloads_only_hubert(self):
        self.sessions.load("a.pth", hubert_variant="chinese")
        sentinel = _FakeGraphCache()
        self.loader.return_value.load.return_value.synthesizer._rvc_cuda_graph_cache = sentinel

        self.sessions.load("a.pth", hubert_variant="base")

        self.assertEqual(self.hubert.call_count, 2)
        self.assertEqual(self.loader.return_value.load.call_count, 1)
        self.assertIs(getattr(self.loader.return_value.load.return_value.synthesizer,
                              "_rvc_cuda_graph_cache", None), sentinel,
                      "只换 hubert 变体时合成器仍复用同一对象，图缓存同样不该被拆")


class TestInferenceCacheEviction(unittest.TestCase):
    def test_evicted_bundle_loses_graph_cache(self):
        from rvc.pipeline.cache import InferenceCache

        cache = InferenceCache()
        bundles = [_FakeBundle() for _ in range(3)]   # synthesizer 槽位只留 2 个
        with mock.patch("rvc.runtime.graph.gc.collect"), \
                mock.patch("rvc.runtime.graph.torch.cuda.is_available", return_value=False):
            for index, bundle in enumerate(bundles):
                bundle.synthesizer._rvc_cuda_graph_cache = _FakeGraphCache()
                cache.set_synthesizer(("cpu", True, f"{index}.pth"), bundle)

        self.assertIsNone(cache.get_synthesizer(("cpu", True, "0.pth")))
        self.assertFalse(hasattr(bundles[0].synthesizer, "_rvc_cuda_graph_cache"),
                         "被淘汰的模型必须显式回收图缓存，不能留给 GC")
        self.assertTrue(hasattr(bundles[2].synthesizer, "_rvc_cuda_graph_cache"))


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


class TestPitchMapBackends(unittest.TestCase):
    """音域映射双后端收敛成一份实现后，两条路径必须仍然等价。"""

    def _f0(self):
        rng = np.random.default_rng(11)
        return np.concatenate([
            np.zeros(5, dtype=np.float32),          # 清音帧
            rng.uniform(50, 900, 120).astype(np.float32),
        ])

    def test_torch_matches_numpy(self):
        from rvc.pipeline.pitch.postprocess import apply_pitch_map

        f0 = self._f0()
        expected = apply_pitch_map(f0, 60, 720, 160, 720)
        got = apply_pitch_map(torch.from_numpy(f0), 60, 720, 160, 720).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_unvoiced_stay_zero_and_output_clamped(self):
        from rvc.pipeline.pitch.postprocess import apply_pitch_map

        f0 = self._f0()
        out = apply_pitch_map(f0, 60, 720, 160, 720)
        self.assertTrue(np.all(out[:5] == 0), "清音帧必须保持 0，不能被映射成有声")
        self.assertGreaterEqual(float(out[5:].min()), 160.0 - 1e-3)
        self.assertLessEqual(float(out[5:].max()), 720.0 + 1e-3)

    def test_invalid_range_returns_input_untouched(self):
        from rvc.pipeline.pitch.postprocess import apply_pitch_map

        f0 = torch.linspace(100, 200, 8)
        self.assertIs(apply_pitch_map(f0, 720, 60, 160, 720), f0)
        self.assertIs(apply_pitch_map(f0, 60, 720, 720, 160), f0)

    def test_constants_cached_per_device(self):
        from rvc.pipeline.pitch import postprocess

        postprocess._pitch_map_tensor_cache.clear()
        f0 = torch.linspace(100, 200, 8)
        postprocess.apply_pitch_map(f0, 60, 720, 160, 720)
        postprocess.apply_pitch_map(f0, 60, 720, 160, 720)
        self.assertEqual(len(postprocess._pitch_map_tensor_cache), 1,
                         "同参数同设备只该有一份常量张量")


if __name__ == "__main__":
    unittest.main()
