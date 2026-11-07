"""训练主循环契约 — epoch 区间、保存节奏、软停止、续训起点、导出路径。

train() 的循环逻辑此前零覆盖：它不碰 GPU 的部分（range 边界、save 触发条件、
stop_requested 语义、返回路径命名）可以用桩替掉 setup/_train_epoch/_save 后
直接跑真实方法锁住。GPU 相关的数值契约另见 test_train_pipeline_contracts。
"""
import tempfile
import unittest
from pathlib import Path

from rvc.core.config import TrainConfig
from rvc.train.trainer import Trainer


class _StepCounter:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _loop_trainer(tmp, *, epochs, save_every=200, start_epoch=1, stop_after=None):
    """搭一个只跑循环、不碰 GPU 的 Trainer，记录每个 epoch 的保存与调度。"""
    exp = Path(tmp) / "exp"
    exp.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(exp_dir=str(exp), epochs=epochs, save_every_epoch=save_every)
    trainer = Trainer(cfg, export_dir=Path(tmp) / "models")
    saved, ran = [], []

    def fake_setup():
        trainer.start_epoch = start_epoch
        trainer.scheduler_g = _StepCounter()
        trainer.scheduler_d = _StepCounter()

    def fake_epoch(epoch):
        ran.append(epoch)
        if stop_after is not None and epoch >= stop_after:
            trainer.stop_requested = True
        return (1.0, 2.0, 3.0, 4.0, 5.0)

    trainer.setup = fake_setup
    trainer._train_epoch = fake_epoch
    trainer._save = lambda epoch: saved.append(epoch)
    trainer.log = lambda message: None
    return trainer, saved, ran


class TestTrainLoop(unittest.TestCase):
    def test_runs_start_epoch_through_epochs_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, saved, ran = _loop_trainer(tmp, epochs=5, start_epoch=3)
            trainer.train()
            self.assertEqual(ran, [3, 4, 5])
            self.assertEqual(trainer.scheduler_g.steps, 3)
            self.assertEqual(trainer.scheduler_d.steps, 3)

    def test_save_cadence_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, saved, ran = _loop_trainer(tmp, epochs=5, save_every=2)
            trainer.train()
            # 2、4 命中间隔，5 是最后一轮必存
            self.assertEqual(saved, [2, 4, 5])

    def test_stop_saves_current_epoch_then_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, saved, ran = _loop_trainer(tmp, epochs=10, save_every=100, stop_after=3)
            out = trainer.train()
            self.assertEqual(ran, [1, 2, 3])
            self.assertEqual(saved, [3], "软停止当轮必须保存")
            self.assertTrue(out.endswith("_e3.pth"))

    def test_stop_before_first_epoch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, saved, ran = _loop_trainer(tmp, epochs=5)
            trainer.stop_requested = True
            with self.assertRaises(RuntimeError):
                trainer.train()
            self.assertEqual(ran, [])
            self.assertEqual(saved, [])

    def test_return_path_is_export_dir_expname_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, saved, ran = _loop_trainer(tmp, epochs=2)
            out = trainer.train()
            self.assertEqual(out, str(Path(tmp) / "models" / "exp_e2.pth"))


if __name__ == "__main__":
    unittest.main()