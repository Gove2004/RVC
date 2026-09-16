"""checkpoint / 导出格式 / spec 缓存的行为基线测试。

覆盖（基线清单 §4.8/§4.9/§4.11 + S8 改动）：
- checkpoint 保存/加载往返，缺键跳过并告警
- 导出模型格式：half 权重、去 enc_q、config 18 元列表、原子写
- 与现有真实导出模型的互操作（键结构一致）
- spec 缓存落 spec_cache/ 目录 + mtime 失效语义
"""
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rvc.runtime.paths import MODELS_DIR, PROJECT_ROOT
from rvc.train.checkpoint import (
    exported_epoch,
    export_model,
    load_checkpoint,
    load_train_json,
    prune_keep_latest,
    save_checkpoint,
)
from rvc.train.data_utils import TextAudioLoaderMultiNSFsid


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.extra = torch.nn.Linear(2, 2)


class TestCheckpointRoundtrip(unittest.TestCase):
    def test_save_load_roundtrip(self):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "G_7.pth")
            save_checkpoint(model, optimizer, 1e-4, 7, path)
            clone = _TinyModel()
            clone.lin.bias.data.fill_(99.0)  # 应被 checkpoint 覆盖回去
            lr, epoch = load_checkpoint(path, clone, optimizer)
            self.assertEqual(epoch, 7)
            self.assertEqual(lr, 1e-4)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(clone.state_dict()[key], value), key)

    def test_mismatched_keys_skipped_with_warning(self):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "G_1.pth")
            save_checkpoint(model, optimizer, 1e-4, 1, path)
            # 形状不同的同名层：应被跳过而不是崩溃
            class _Other(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = torch.nn.Linear(8, 4)
            other = _Other()
            with self.assertLogs("rvc.train.checkpoint", level="WARNING"):
                lr, epoch = load_checkpoint(path, other)
            self.assertEqual(epoch, 1)


class TestPruneAndEpoch(unittest.TestCase):
    def test_exported_epoch_and_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for epoch in (3, 10, 25):
                (d / f"m_e{epoch}.pth").write_bytes(b"x")
            (d / "unrelated.pth").write_bytes(b"x")
            self.assertEqual(exported_epoch(d / "m_e25.pth"), 25)
            removed = prune_keep_latest(d, "m_e*.pth", 1, epoch_of=exported_epoch)
            self.assertEqual([p.name for p in removed], ["m_e3.pth", "m_e10.pth"])
            self.assertEqual([p.name for p in d.glob("m_e*.pth")], ["m_e25.pth"])


class TestExportFormat(unittest.TestCase):
    def test_export_model_format(self):
        json_config = load_train_json(40000)
        state = {k: torch.randn(4, 4) for k in ("enc_q.weight", "enc_q.bias", "dec.weight")}
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "exp_e5.pth")
            export_model(state, 40000, json_config, 5, out)
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
        self.assertEqual(set(ckpt.keys()), {"weight", "config", "info", "sr", "f0", "version"})
        self.assertNotIn("enc_q.weight", ckpt["weight"])
        self.assertNotIn("enc_q.bias", ckpt["weight"])
        self.assertIn("dec.weight", ckpt["weight"])
        for value in ckpt["weight"].values():
            self.assertEqual(value.dtype, torch.float16)
        self.assertEqual(len(ckpt["config"]), 18)
        self.assertEqual(ckpt["sr"], "40k")
        self.assertEqual(ckpt["f0"], 1)
        self.assertEqual(ckpt["version"], "v2")
        self.assertEqual(ckpt["info"], "exp")

    def test_real_exported_model_interop(self):
        """现有真实导出模型（旧代码产生）的结构与新 export_model 完全同构。"""
        models = sorted(MODELS_DIR.glob("*.pth"))
        if not models:
            self.skipTest("assets/models/ 无导出模型")
        ckpt = torch.load(str(models[0]), map_location="cpu", weights_only=False)
        self.assertEqual(set(ckpt.keys()), {"weight", "config", "info", "sr", "f0", "version"})
        self.assertNotIn("enc_q.weight", ckpt["weight"])
        self.assertEqual(len(ckpt["config"]), 18)


class TestSpecCache(unittest.TestCase):
    """A6：spec 缓存写 {exp}/spec_cache/，mtime 失效语义保留。"""

    SR = 16000
    DATA_CFG = {
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
    }

    def _make_fixture(self, exp: Path):
        gt = exp / "0_gt_wavs"
        gt.mkdir(parents=True)
        t = np.arange(self.SR, dtype=np.float32) / self.SR
        wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        from rvc.io.wav_file import write_wav

        write_wav(str(gt / "1_0.wav"), wav, self.SR, subtype="FLOAT")
        frames = 60
        np.save(gt / "1_0.feat.npy", np.zeros((frames, 768), dtype=np.float32))
        np.save(gt / "1_0.f0.npy", np.full(frames, 100, dtype=np.int64))
        np.save(gt / "1_0.f0nsf.npy", np.full(frames, 220.0, dtype=np.float32))
        filelist = exp / "filelist.txt"
        filelist.write_text(
            f"{gt / '1_0.wav'}|{gt / '1_0.feat.npy'}|{gt / '1_0.f0.npy'}|{gt / '1_0.f0nsf.npy'}|0",
            encoding="utf-8",
        )

    def test_cache_written_to_spec_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            self._make_fixture(exp)
            cache_dir = exp / "spec_cache"
            loader = TextAudioLoaderMultiNSFsid(
                str(exp / "filelist.txt"), self.DATA_CFG, spec_cache_dir=cache_dir
            )
            self.assertEqual(len(loader), 1)
            phone, pitch, pitchf, spec, wav, sid = loader[0]
            self.assertEqual(spec.ndim, 2)  # [freq, frames]
            self.assertEqual(spec.shape[0], self.DATA_CFG["filter_length"] // 2 + 1)
            self.assertEqual(int(sid), 0)
            caches = list(cache_dir.glob("*.spec.pt"))
            self.assertEqual(len(caches), 1, "缓存应写在 spec_cache/ 目录")
            self.assertTrue(caches[0].name.startswith("1_0.sr16000"))
            # 切片目录内不得再有 spec 缓存
            self.assertEqual(list((exp / "0_gt_wavs").glob("*.spec.pt")), [])

            # mtime 失效语义：更新 wav 后缓存重建
            first_mtime = caches[0].stat().st_mtime
            import os
            import time

            wav_path = exp / "0_gt_wavs" / "1_0.wav"
            future = time.time() + 10
            os.utime(wav_path, (future, future))
            loader[0]
            self.assertGreater(caches[0].stat().st_mtime, first_mtime, "wav 更新后缓存应重建")


if __name__ == "__main__":
    unittest.main()
