"""训练链路契约 — 这四条都是「整条链路必崩」级别，且单测当时全都放过了。

共同点：每处都只在真正把预处理→F0→特征→训练→导出跑一遍时才暴露，
既有的 187 项单测全过也发现不了。所以这里锁的不是算法正确性，而是
「这条路径能跑通」这个最低契约。
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rvc.train.checkpoint import latest_checkpoint_path, save_checkpoint, save_checkpoint_pair
from rvc.train.dataset import TextAudioCollateMultiNSFsid, TrainBatch, TrainSample


def _sample(phone_len: int, spec_len: int, wav_len: int, feat_dim: int = 4,
            spec_channels: int = 3, sid: int = 0) -> TrainSample:
    return TrainSample(
        phone=torch.zeros(phone_len, feat_dim),
        pitch=torch.zeros(phone_len, dtype=torch.long),
        pitchf=torch.zeros(phone_len),
        spec=torch.zeros(spec_channels, spec_len),
        wav=torch.zeros(wav_len),
        sid=sid,
    )


class TestCollateProducesTrainBatch(unittest.TestCase):
    """collate 的返回值必须能被 TrainBatch 原样接住。

    曾经多传了第 9 个字段 wav_lengths，而 TrainBatch 只有 8 个字段——
    DataLoader 取第一个 batch 就 TypeError。TrainBatch 必须保持 NamedTuple，
    不能为了迁就调用方改成普通容器（钉页只对 namedtuple 生效）。
    """

    def test_padded_batch_field_shapes(self):
        batch = TextAudioCollateMultiNSFsid()([
            _sample(phone_len=5, spec_len=10, wav_len=1600, sid=0),
            _sample(phone_len=3, spec_len=7, wav_len=900, sid=1),
        ])

        self.assertIsInstance(batch, TrainBatch)
        self.assertEqual(len(batch), 8, "TrainBatch 字段数变了就要同步改 collate")
        self.assertEqual(batch.phone.shape, (2, 5, 4), "phone 按最长序列补零")
        self.assertEqual(batch.spec.shape, (2, 3, 10), "spec 按最长帧数补零")
        self.assertEqual(batch.wav.shape, (2, 1600), "wav 按最长采样数补零")
        self.assertEqual(batch.phone_lengths.tolist()[0], 5)
        self.assertEqual(batch.spec_lengths.tolist()[0], 10)
        self.assertEqual(batch.sid.tolist(), [0, 1])

    def test_short_sample_padding_is_zeroed(self):
        batch = TextAudioCollateMultiNSFsid()([
            _sample(phone_len=5, spec_len=10, wav_len=1600),
            _sample(phone_len=3, spec_len=7, wav_len=900),
        ])
        # 排序按 spec 长度降序，第二个 batch 位置是短样本
        self.assertTrue(torch.all(batch.phone[1, 3:] == 0), "超出真实长度的部分必须是零")
        self.assertTrue(torch.all(batch.wav[1, 900:] == 0))

    def test_unpacking_matches_trainer(self):
        """训练循环按 8 个名字解包，字段顺序一旦变动这里先炸。"""
        batch = TextAudioCollateMultiNSFsid()([_sample(4, 8, 100), _sample(2, 6, 50)])
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wav, sid = batch
        self.assertEqual(phone.shape[0], 2)
        self.assertEqual(wav.shape[0], 2)
        self.assertEqual(sid.shape[0], 2)


class TestCheckpointPairWritesAndFinds(unittest.TestCase):
    """成对保存 + 找最新 checkpoint：训练 setup 与每次保存都要走这条路径。

    曾经用了 os.replace / os.path.join 却没 import os，两个方向都抛
    NameError——即「一次 checkpoint 都存不下来，续训也找不到」。
    """

    def _pair(self, tmp: str, epoch: int):
        g, d = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        g_opt = torch.optim.AdamW(g.parameters(), lr=1e-4)
        d_opt = torch.optim.AdamW(d.parameters(), lr=1e-4)
        g_path = os.path.join(tmp, f"G_{epoch}.pth")
        d_path = os.path.join(tmp, f"D_{epoch}.pth")
        save_checkpoint_pair(g, g_opt, d, d_opt, 1e-4, epoch, g_path, d_path)
        return g_path, d_path

    def test_pair_written_without_temp_leftovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            g_path, d_path = self._pair(tmp, 1)
            self.assertTrue(os.path.exists(g_path))
            self.assertTrue(os.path.exists(d_path))
            self.assertEqual(
                [p for p in os.listdir(tmp) if p.endswith(".pairtmp")], [],
                "两段 os.replace 之后不许残留 .pairtmp")

    def test_latest_checkpoint_path_picks_highest_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(latest_checkpoint_path(tmp, "G"), "空目录应返回 None")
            for epoch in (1, 2, 10):
                self._pair(tmp, epoch)
            # 按数值而非字典序：G_10 必须赢过 G_2
            self.assertEqual(os.path.basename(latest_checkpoint_path(tmp, "G")), "G_10.pth")


@unittest.skipUnless(torch.cuda.is_available(), "GradScaler(enabled=True) 需要 CUDA 才能构造 _scale")
class TestGradScalerStateSurvivesResume(unittest.TestCase):
    """fp16 的损失缩放倍率必须能跨续训保留。

    默认 init_scale=65536 对 fp16 上限 65504 没有余量，本模型梯度量级
    O(1)~O(10²)，实测每 batch 折半一次、约 7 个 batch 才稳定到 256。
    不存进 checkpoint 就每次续训从 65536 重新预热：白烧前几个 batch，
    还会先看到一段 inf 和虚高的 loss。

    必须用 enabled=True 的 scaler 来测：禁用态的 state_dict() 恒为空、
    get_scale() 恒为 1.0，拿它测往返等于什么都没测。
    """

    def _scaler(self, scale: float = 256.0):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler._scale = torch.tensor(scale, device="cuda")
        return scaler

    def test_scaler_state_round_trips(self):
        from rvc.train.checkpoint import load_checkpoint

        scaler = self._scaler(256.0)

        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "G_1.pth")
            save_checkpoint(model, opt, 1e-4, 1, path, scaler)

            restored = torch.amp.GradScaler("cuda", enabled=True)
            self.assertEqual(float(restored.get_scale()), 65536.0, "新实例应是默认初值")
            load_checkpoint(path, torch.nn.Linear(2, 2), None, restored)
            self.assertEqual(float(restored.get_scale()), 256.0, "续训后不应重新预热")

    def test_old_checkpoint_without_scaler_still_loads(self):
        """旧 checkpoint 没有 scaler 键，续训必须照常工作（不能 KeyError）。"""
        from rvc.train.checkpoint import load_checkpoint

        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "G_1.pth")
            torch.save({
                "model": model.state_dict(), "iteration": 1,
                "optimizer": None, "learning_rate": 1e-4,
            }, path)

            scaler = self._scaler(512.0)
            lr, epoch = load_checkpoint(path, torch.nn.Linear(2, 2), None, scaler)
            self.assertEqual((lr, epoch), (1e-4, 1))
            self.assertEqual(float(scaler.get_scale()), 512.0, "无 scaler 键时应保持原状")

    def test_pair_saves_same_scaler_copy(self):
        """G/D 成对保存时，两个文件里应是同一个 scaler 的同一份状态。"""
        scaler = self._scaler(128.0)

        with tempfile.TemporaryDirectory() as tmp:
            g, d = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            g_path = os.path.join(tmp, "G_2.pth")
            d_path = os.path.join(tmp, "D_2.pth")
            save_checkpoint_pair(
                g, torch.optim.AdamW(g.parameters(), lr=1e-4),
                d, torch.optim.AdamW(d.parameters(), lr=1e-4),
                1e-4, 2, g_path, d_path, scaler,
            )
            for path in (g_path, d_path):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                self.assertIsNotNone(saved["scaler"], path)
                self.assertEqual(float(saved["scaler"]["scale"]), 128.0, path)


class TestGradScalerInitScale(unittest.TestCase):
    """GradScaler 初始倍率不能沿用 torch 默认的 65536。

    实测（batch_size=2，每 epoch 6 batch，加载 f0G40k 底模）：
      65536 起步时前 7 个 batch 全部溢出折半，D 到第 3 批、G 到第 8 批才第一次
      真正 optimizer.step()——也就是第 1 个 epoch 生成器一次都没更新，还会触发
      torch 的「lr_scheduler.step() 早于 optimizer.step()」警告。
      改成 256 起步后 D 第 0 批、G 第 1 批就更新，警告消失，稳态倍率同样是 128
      → 数值行为不变，只是省掉了折半过程。
    """

    def test_default_is_not_torch_default(self):
        from rvc.core.constants import TRAIN_GRAD_SCALER_INIT_SCALE_DEFAULT as D
        from rvc.train.trainer import grad_scaler_init_scale

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RVC_TRAIN_INIT_SCALE", None)
            self.assertEqual(grad_scaler_init_scale(), D)
        self.assertNotEqual(D, 2 ** 16, "不要退回 torch 默认值")
        self.assertGreater(D, 0)
        # fp16 上限 65504，初始倍率必须有余量
        self.assertLessEqual(D, 65504)

    def test_env_override(self):
        from rvc.train.trainer import grad_scaler_init_scale

        with mock.patch.dict(os.environ, {"RVC_TRAIN_INIT_SCALE": "1024"}):
            self.assertEqual(grad_scaler_init_scale(), 1024)

    def test_invalid_env_falls_back_to_default(self):
        from rvc.core.constants import TRAIN_GRAD_SCALER_INIT_SCALE_DEFAULT as D
        from rvc.train.trainer import grad_scaler_init_scale

        for bad in ("abc", "", "   ", "1.5"):
            with mock.patch.dict(os.environ, {"RVC_TRAIN_INIT_SCALE": bad}):
                self.assertEqual(grad_scaler_init_scale(), D, bad)

    def test_zero_and_negative_clamped_to_one(self):
        """scaler 倍率必须为正，0 会让整个 AMP 失效。"""
        from rvc.train.trainer import grad_scaler_init_scale

        for bad in ("0", "-4096"):
            with mock.patch.dict(os.environ, {"RVC_TRAIN_INIT_SCALE": bad}):
                self.assertEqual(grad_scaler_init_scale(), 1, bad)


class TestLrScheduleAcrossResume(unittest.TestCase):
    """学习率轨迹：epoch N 必须用 gamma^(N-1)，续训不许额外衰减一次。

    trainer.py 用 `last_epoch = start_epoch - 2` 重建 ExponentialLR。这个 -2 很容易
    被误认为 off-by-one 而"顺手修正"：ExponentialLR.get_lr() 是纯乘性（在当前 lr
    上再乘 gamma），看着像会多衰减一次。实测（含续训）偏差为 0，即 -2 是对的——
    last_epoch>=0 时构造走 closed-form 路径，用 initial_lr*gamma^last_epoch 重算，
    不做累乘。所以别改这个 -2，改之前先让本文件失败。
    """

    GAMMA = 0.999875  # 40k/48k 配置里的 lr_decay
    BASE = 1e-4

    def _fresh(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=self.BASE)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.GAMMA, last_epoch=-1)
        return opt, sch

    def test_fresh_epoch_uses_gamma_pow_n_minus_1(self):
        opt, sch = self._fresh()
        for epoch in range(1, 7):
            expected = self.BASE * self.GAMMA ** (epoch - 1)
            self.assertAlmostEqual(
                opt.param_groups[0]["lr"], expected, places=15,
                msg=f"epoch {epoch} 的 LR 应是 gamma^{epoch - 1}")
            opt.step()
            sch.step()

    def test_resume_does_not_decay_an_extra_time(self):
        """在 epoch 5 末存 checkpoint，续训后 epoch 6 仍是 gamma^5。"""
        opt, sch = self._fresh()
        for _ in range(5):
            opt.step()
            sch.step()
        lr_at_save = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_at_save, self.BASE * self.GAMMA ** 5, places=15)
        self.assertIn("initial_lr", opt.param_groups[0], "initial_lr 必须随 optimizer 状态落盘")

        saved = opt.state_dict()

        # 续训：恢复 optimizer 状态，再按 trainer 的公式重建 scheduler
        opt2, _ = self._fresh()
        opt2.load_state_dict(saved)
        torch.optim.lr_scheduler.ExponentialLR(
            opt2, gamma=self.GAMMA, last_epoch=6 - 2)

        self.assertAlmostEqual(
            opt2.param_groups[0]["lr"], lr_at_save, places=15,
            msg="续训后的 LR 必须与 checkpoint 时一致；变小了说明多衰减了一次")
        self.assertAlmostEqual(
            opt2.param_groups[0]["lr"], self.BASE * self.GAMMA ** 5, places=15)

    def test_resume_continues_from_next_epoch(self):
        """续训后走完一个 epoch，epoch 7 应拿到 gamma^6（衔接不跳步）。"""
        opt, sch = self._fresh()
        for _ in range(5):
            opt.step()
            sch.step()
        saved = opt.state_dict()

        opt2, _ = self._fresh()
        opt2.load_state_dict(saved)
        sch2 = torch.optim.lr_scheduler.ExponentialLR(
            opt2, gamma=self.GAMMA, last_epoch=6 - 2)
        opt2.step()
        sch2.step()
        self.assertAlmostEqual(
            opt2.param_groups[0]["lr"], self.BASE * self.GAMMA ** 6, places=15)


class TestSineGenStaysTrainable(unittest.TestCase):
    """SineGen 的输出要喂给可训练的 SourceModuleHnNSF.l_linear。

    它曾经无条件 `with torch.inference_mode()`：fp32 下 backward 直接抛
    "Inference tensors cannot be saved for backward"，fp16 下被 autocast
    掩盖成静默断链（谐波支路拿不到梯度）。推理侧调用方本就在
    inference_mode 里，所以「不求梯度才进」既修训练又不损推理速度。
    """

    def setUp(self):
        from rvc.models.synthesizer_decoder import SineGen
        # 与本项目的实际构造一致：SourceModuleHnNSF 固定 harmonic_num=0（dim=1）
        self.sine_gen = SineGen(40000, harmonic_num=0)
        self.f0 = torch.full((1, 20), 200.0)

    def test_output_is_not_inference_tensor_when_grad_enabled(self):
        sine_waves, uv, noise = self.sine_gen(self.f0, upp=480)
        self.assertFalse(sine_waves.is_inference(),
                         "求梯度时必须是普通张量，否则下游可训练层无法反传")
        self.assertEqual(sine_waves.shape, (1, 20 * 480, 1))

    def test_gradient_reaches_downstream_linear(self):
        sine_waves, _, _ = self.sine_gen(self.f0, upp=480)
        projection = torch.nn.Linear(sine_waves.shape[-1], 1)
        projection(sine_waves).sum().backward()
        self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(torch.isfinite(projection.weight.grad).all())

    def test_still_inference_mode_under_no_grad(self):
        """推理路径（调用方已进 inference_mode）必须保持原优化，别退化成普通张量。"""
        with torch.inference_mode():
            sine_waves, _, _ = self.sine_gen(self.f0, upp=480)
        self.assertTrue(sine_waves.is_inference(),
                        "推理侧仍要拿到 inference tensor，否则丢性能")


class TestHubertCacheEviction(unittest.TestCase):
    """训练侧 HuBERT 缓存用 LRUCache：写入方法是 insert，返回被淘汰值作回收钩子。

    曾经调了不存在的 .set()，特征提取第一步就 AttributeError。
    """

    def test_insert_evicts_and_purges_graph_cache(self):
        from rvc.runtime.lru import LRUCache
        from rvc.train import extract_feature

        cache = LRUCache(1)
        with mock.patch.object(extract_feature, "_hubert_cache", cache), \
                mock.patch.object(extract_feature, "load_hubert",
                                  side_effect=lambda cfg, variant: SimpleNamespace(variant=variant)), \
                mock.patch.object(extract_feature, "purge_cuda_graphs") as purge:
            first = extract_feature.HuBERTExtractor(hubert="base")
            self.assertIs(cache.get(("cuda:0", True, "base")), first.model)
            purge.assert_not_called()

            extract_feature.HuBERTExtractor(hubert="chinese")
            purge.assert_called_once_with(first.model)

    def test_same_config_reuses_cached_model(self):
        from rvc.runtime.lru import LRUCache
        from rvc.train import extract_feature

        cache = LRUCache(2)
        with mock.patch.object(extract_feature, "_hubert_cache", cache), \
                mock.patch.object(extract_feature, "load_hubert",
                                  side_effect=lambda cfg, variant: SimpleNamespace(variant=variant)) as load:
            first = extract_feature.HuBERTExtractor(hubert="base")
            second = extract_feature.HuBERTExtractor(hubert="base")
        self.assertIs(first.model, second.model, "同配置必须复用，别重复加载 ~1GB 权重")
        self.assertEqual(load.call_count, 1)


class TestHubertCacheClearReleases(unittest.TestCase):
    """clear() 只丢引用不回收值本身，调用方必须先 release。

    特征提取完成后、正式训练前会 clear_hubert_cache() 腾出 ~1GB。若只 clear()
    不回收图缓存，被丢掉的 HuBERT 持有的静态张量不会随引用消失归还，训练紧接着
    捕获自己的图时可能复用同一批地址。
    """

    def test_clear_purges_every_cached_model(self):
        from rvc.runtime.lru import LRUCache
        from rvc.train import extract_feature

        cache = LRUCache(2)
        with mock.patch.object(extract_feature, "_hubert_cache", cache), \
                mock.patch.object(extract_feature, "load_hubert",
                                  side_effect=lambda cfg, variant: SimpleNamespace(variant=variant)), \
                mock.patch.object(extract_feature, "purge_cuda_graphs") as purge, \
                mock.patch("torch.cuda.empty_cache"):
            base = extract_feature.HuBERTExtractor(hubert="base").model
            chinese = extract_feature.HuBERTExtractor(hubert="chinese").model
            purge.reset_mock()

            extract_feature.clear_hubert_cache()

        self.assertEqual(purge.call_count, 2, "缓存里每个模型都要回收")
        purged = {id(c.args[0]) for c in purge.call_args_list}
        self.assertEqual(purged, {id(base), id(chinese)})
        self.assertEqual(cache.values(), [], "回收之后才允许清空")

    def test_clear_on_empty_cache_is_noop(self):
        from rvc.runtime.lru import LRUCache
        from rvc.train import extract_feature

        with mock.patch.object(extract_feature, "_hubert_cache", LRUCache(2)), \
                mock.patch.object(extract_feature, "purge_cuda_graphs") as purge, \
                mock.patch("torch.cuda.empty_cache"):
            extract_feature.clear_hubert_cache()
        purge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
