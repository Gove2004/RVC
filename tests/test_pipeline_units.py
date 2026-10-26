"""管线单元契约 — 三处「静默算错」的回归锁定。

- engine.load_model 快路径必须同时比对 pth 与 hubert 变体（否则离线页选了 base
  却沿用 chinese 特征器，音色发闷且无告警）
- formant 重采样核缓存的键必须含两侧频率（upp_res 可在不同 sr 下撞出同值）
- normalize_f0_to_coarse 的 torch 分支必须与 numpy 分支逐值等价，且不使用
  布尔掩码索引（数据依赖形状 → 每块强制同步 + 无法进 CUDA Graph）
"""
import math
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from rvc.core.constants import HUBERT_LOOKAHEAD_BLOCKS, HUBERT_LOOKAHEAD_DEFAULT
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
        pipeline = mock.Mock(model_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor:
            self.assertEqual(engine.load_model("m.pth", hubert="chinese"), 40000)
            ctor.assert_not_called()

    def test_same_pth_different_variant_reloads(self):
        """实时页用 chinese 加载后，离线页切 base 必须真的重新加载。"""
        pipeline = mock.Mock(model_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor, \
                mock.patch("torch.cuda.empty_cache"), mock.patch("gc.collect"):
            ctor.return_value.model_sr = 40000
            engine.load_model("m.pth", hubert="base")
            ctor.assert_called_once()
            self.assertEqual(ctor.call_args.kwargs["hubert"], "base")

    def test_force_reloads_same_pth(self):
        pipeline = mock.Mock(model_sr=40000, hubert_variant="chinese")
        engine = _bare_engine(pipeline, "m.pth")
        with mock.patch("rvc.pipeline.pipeline.InferencePipeline") as ctor, \
                mock.patch("torch.cuda.empty_cache"), mock.patch("gc.collect"):
            ctor.return_value.model_sr = 40000
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
        self.model_sr = 40000
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
        # reloaded=False → 不重复 remove_weight_norm
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


class TestFormantRuntimeAdjust(unittest.TestCase):
    """性别因子（formant）可在变声运行中调整。

    formant 经 return_length2 决定合成器输出形状，每换一个整数帧数就是另一张图；
    量程内的全部档位由 prewarm_formant 在流启动前预捕获（实测每档 0.19~0.30s，
    全量程 ~2s），所以运行中切换只是回放另一张图，不阻塞音频回调。

    配置（runtime_params）与运行态（state）之间没有隐式同步：只有显式的
    engine.set_formant 才会改运行态，避免"改了配置就悄悄生效"这条通路。
    """

    def _runner(self, formant: float = 0.0):
        from rvc.core.config import InferenceParams
        from rvc.pipeline.context import InferenceContext
        from rvc.pipeline.state import EngineState
        from rvc.streaming.runner import InferenceRunner

        params = InferenceParams()
        params.voice.formant = formant
        pipeline = mock.Mock()
        pipeline.state = EngineState(device="cpu", is_half=False)
        pipeline.ctx = InferenceContext()
        runner = InferenceRunner(pipeline, params)
        runner.init_processing(40000, 0.25, 0.05, 0.05, 1, 40000)
        return runner, params

    def test_hubert_lookahead_matches_configured_blocks(self):
        """滚动窗口在消费区后保留 `HUBERT_LOOKAHEAD_BLOCKS` 个完整输入块。"""
        runner, _ = self._runner(0.0)
        state = runner.state
        base_window = (
            state.extra_samples
            + state.crossfade_samples
            + state.sola_search_samples
            + state.block_samples
        )
        self.assertEqual(state.lookahead_samples, HUBERT_LOOKAHEAD_BLOCKS * state.block_samples)
        self.assertEqual(state.input_wav_work.numel(), base_window + state.lookahead_samples)

    def test_hubert_lookahead_tracks_shorter_block(self):
        runner, _ = self._runner(0.0)
        runner.init_processing(40000, 0.10, 0.05, 0.05, 1, 40000)
        self.assertEqual(runner.state.block_samples, 4000)
        self.assertEqual(runner.state.lookahead_samples, HUBERT_LOOKAHEAD_BLOCKS * 4000)

    def test_lookahead_defaults_to_off(self):
        """默认不加延迟：前瞻收益只在特征层，听感层没兑现，不值得 +1 块延迟。"""
        self.assertEqual(HUBERT_LOOKAHEAD_DEFAULT, 0)

    def test_lookahead_env_override_is_bounded(self):
        from rvc.core.constants import _env_nonnegative_int

        name = "RVC_HUBERT_LOOKAHEAD_BLOCKS"
        with mock.patch.dict("os.environ", {name: "2"}):
            self.assertEqual(_env_nonnegative_int(name, 0), 2)
        for bad in ("-3", "abc", "", "  "):
            with mock.patch.dict("os.environ", {name: bad}):
                self.assertEqual(_env_nonnegative_int(name, 0), 0)

    def test_engine_lookahead_telemetry_is_initialized(self):
        from rvc.core.config import InferenceParams
        from rvc.streaming.engine import VoiceEngine

        engine = VoiceEngine(InferenceParams())
        self.assertEqual(engine.lookahead_ms, 0.0)

    def test_factor_initialized_from_config(self):
        runner, _ = self._runner(1.5)
        self.assertAlmostEqual(runner.state.formant_factor, 2 ** (1.5 / 12), places=6)

    def test_config_change_alone_does_not_touch_state(self):
        """拖滑块只写 runtime_params；运行态由 engine.set_formant 显式下发。"""
        runner, params = self._runner(1.5)
        before = runner.state.formant_factor
        params.voice.formant = -2.5
        self.assertEqual(runner.state.formant_factor, before)

    def test_set_formant_factor_updates_state(self):
        runner, _ = self._runner(0.0)
        runner.set_formant_factor(2 ** (2.0 / 12))
        self.assertAlmostEqual(runner.state.formant_factor, 2 ** (2.0 / 12), places=9)

    def test_stage_formant_uses_quantized_ratio(self):
        """重采样比例取量化后的 formant_ratio，而不是原始 formant_factor。"""
        runner, _ = self._runner(0.0)
        state = runner.state
        state.formant_ratio = 36 / 31
        expected = (state.block_samples + state.sola_buffer_samples
                    + state.sola_search_samples)
        infer = torch.zeros(36 * (state.block_samples // 31))
        out = runner._stage_formant(infer)
        self.assertEqual(out.shape[0], expected, "formant 处理后必须对齐到固定块长")

    def test_stage_formant_is_identity_when_ratio_is_one(self):
        runner, _ = self._runner(0.0)
        state = runner.state
        state.formant_factor = 1.2          # 原始因子不为 1
        state.formant_ratio = 1.0           # 但量化后落回基准帧数
        expected = (state.block_samples + state.sola_buffer_samples
                    + state.sola_search_samples)
        infer = torch.zeros(expected)
        self.assertIs(runner._stage_formant(infer), infer)


class TestFormantDoesNotShiftPitch(unittest.TestCase):
    """性别因子只该移动共振峰，不能顺带改音高。

    合成器按 return_length2 帧合成时会把谐波源 interpolate 到 n*upp，这一步
    的时长拉伸本身就把搬回 return_length 帧时的升频抵掉了；若再多乘一次
    return_length2/return_length，音高会被抬一遍（实测 formant=+2.5 时
    396Hz → 460Hz）。所以 pitchf 必须原样返回。
    """

    def _pitchf_for(self, ratio: float) -> torch.Tensor:
        from rvc.core.config import InferenceParams
        from rvc.pipeline.context import InferenceContext
        from rvc.pipeline.pipeline import InferencePipeline
        from rvc.pipeline.state import EngineState

        pipe = InferencePipeline.__new__(InferencePipeline)
        pipe.state = EngineState(device="cpu", is_half=False)
        pipe.state.use_f0 = True
        pipe.state.return_length = 31
        pipe.state.formant_ratio = ratio
        pipe.ctx = InferenceContext()
        pipe.ctx.config = InferenceParams()
        pipe.ctx.f0_raw = torch.full((1, 16), 200.0)
        _, pitchf = pipe._stage_postprocess_f0(pipe.ctx)
        return pitchf

    def test_pitchf_is_independent_of_formant(self):
        base = self._pitchf_for(1.0)
        shifted = self._pitchf_for(36 / 31)
        torch.testing.assert_close(shifted, base,
                                   msg="formant 比例不许进入 pitchf，否则音高会跟着走")


class TestEngineSetFormant(unittest.TestCase):
    """engine.set_formant 是运行态的唯一写入点（GUI 经 controller 调用）。"""

    def test_running_engine_receives_factor(self):
        engine = _bare_engine(mock.Mock(), "x.pth")
        engine._runner = mock.Mock()
        engine.set_formant(2.0)
        factor = engine._runner.set_formant_factor.call_args[0][0]
        self.assertAlmostEqual(factor, 2 ** (2 / 12), places=9)

    def test_not_running_is_noop(self):
        engine = _bare_engine(mock.Mock(), "x.pth")
        engine._runner = None
        engine.set_formant(2.0)  # 未启动：不抛、不构造引擎


class TestRunnerPrewarmWiring(unittest.TestCase):
    """只有实时的 runner 需要预捕获 formant 图；离线一次转换只用一个档位。"""

    def _make(self, **kwargs):
        engine = _bare_engine(mock.Mock(model_sr=40000), "x.pth")
        engine.runtime_params = mock.Mock()
        with mock.patch("rvc.streaming.engine.InferenceRunner") as ctor, \
                mock.patch("torch.cuda.is_available", return_value=False):
            engine._make_runner(40000, 1, 0.25, 0.05, 2.5, 40000, **kwargs)
            return ctor.return_value

    def test_realtime_prewarms_after_warmup(self):
        runner = self._make(reset_buffers=True, prewarm_formant=True)
        runner.warmup.assert_called_once()
        runner.prewarm_formant.assert_called_once()
        args = runner.prewarm_formant.call_args[0]
        from rvc.core.config import FORMANT_MAX, FORMANT_MIN
        self.assertEqual(args, (FORMANT_MIN, FORMANT_MAX))

    def test_offline_skips_prewarm(self):
        runner = self._make(reset_buffers=False)
        self.assertFalse(runner.prewarm_formant.called)


class TestFormantPrewarmCoverage(unittest.TestCase):
    """预捕获必须覆盖滑条上的每一个位置，否则拖动时仍会在回调线程补捕获。"""

    def test_quantized_frames_and_ratio_agree(self):
        from rvc.pipeline.synthesis import formant_frames
        for factor in (0.866, 1.0, 1.1547):
            frames, ratio = formant_frames(factor, 31)
            self.assertEqual(frames, max(1, math.ceil(31 * factor)))
            self.assertAlmostEqual(ratio, frames / 31, places=12)

    def test_every_slider_position_hits_a_prewarmed_slot(self):
        from rvc.core.config import FORMANT_MAX, FORMANT_MIN
        from rvc.pipeline.synthesis import formant_frames

        return_length = 31
        steps = 101  # 滑条 -2.5..2.5，步长 0.05
        positions = [FORMANT_MIN + i * (FORMANT_MAX - FORMANT_MIN) / (steps - 1)
                     for i in range(steps)]
        needed = {formant_frames(2 ** (f / 12), return_length)[0] for f in positions}
        lo = math.ceil(return_length * 2 ** (FORMANT_MIN / 12))
        hi = math.ceil(return_length * 2 ** (FORMANT_MAX / 12))
        self.assertLessEqual(needed, set(range(lo, hi + 1)),
                             "预捕获的量程必须盖住滑条能取到的全部档位")
        self.assertGreater(len(needed), 1, "量程两端不该落在同一张图上")


class TestInputPitchTelemetry(unittest.TestCase):
    """音高遥测必须是「读缓存」，不能由 GUI 触发 GPU 同步。

    每读一次 `.item()` 就是一次队列排空；GUI 想刷多快就得付出多少同步代价。
    数值改由音频线程在写完主输出（队列已排空）时取回成 float，GUI 只读它。
    """

    def test_reads_cached_float(self):
        engine = _bare_engine(mock.Mock(), "x.pth")
        engine._runner = mock.Mock()
        engine._runner.state.input_pitch_hz = 123.4
        self.assertEqual(engine.input_pitch, 123.4)

    def test_no_runner_returns_zero(self):
        engine = _bare_engine(mock.Mock(), "x.pth")
        engine._runner = None
        self.assertEqual(engine.input_pitch, 0.0)


if __name__ == "__main__":
    unittest.main()
