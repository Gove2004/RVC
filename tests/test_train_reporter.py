"""训练遥测回调工厂契约 — _make_train_reporter 从 step_train 抽出后的行为锁。

锁的是 loss 累计/重置与日志路由这些纯逻辑（不碰 CUDA）：on_loss 按 batch==1
重置、五项 loss 累加；on_epoch 报均值后清零；on_trainer_log 把 epoch 行只落盘。
"""
import unittest

from rvc.train.cli.steps import _make_train_reporter


class _FakeLog:
    def __init__(self):
        self.logged = []      # (msg, level)
        self.file_only_msgs = []

    def log(self, msg, level=None):
        self.logged.append((msg, level))

    def file_only(self, msg):
        self.file_only_msgs.append(msg)


def _reporter(device="cpu"):
    log = _FakeLog()
    on_epoch, on_trainer_log, on_loss, on_batch, state = _make_train_reporter(
        log, {"device": device})
    return log, on_epoch, on_trainer_log, on_loss, on_batch, state


def _loss(batch, value=1.0):
    return {"batch": batch, "loss_d": value, "loss_g": value, "loss_mel": value,
            "loss_kl": value, "loss_fm": value}


class TestLossAccumulation(unittest.TestCase):
    def test_accumulates_five_losses_and_count(self):
        _, _, _, on_loss, _, state = _reporter()
        on_loss(_loss(batch=2, value=2.0))
        on_loss(_loss(batch=3, value=4.0))
        self.assertEqual(state["count"], 2)
        self.assertEqual(state["sum"]["d"], 6.0)
        self.assertEqual(state["sum"]["mel"], 6.0)
        self.assertEqual(state["sum"]["fm"], 6.0)

    def test_batch_one_resets_prior_accumulation(self):
        _, _, _, on_loss, _, state = _reporter()
        on_loss(_loss(batch=5, value=10.0))
        on_loss(_loss(batch=1, value=3.0))  # 新 epoch 首个 batch → 清零后再加
        self.assertEqual(state["count"], 1)
        self.assertEqual(state["sum"]["g"], 3.0)

    def test_on_batch_counts_batches(self):
        _, _, _, _, on_batch, state = _reporter()
        on_batch(1, 0, 10)
        on_batch(1, 1, 10)
        self.assertEqual(state["batches"], 2)


class TestEpochReport(unittest.TestCase):
    def test_epoch_logs_average_then_resets(self):
        log, on_epoch, _, on_loss, _, state = _reporter()
        on_loss(_loss(batch=2, value=2.0))
        on_loss(_loss(batch=3, value=4.0))
        on_epoch(epoch=1, total=10)
        epoch_lines = [m for m, lvl in log.logged if lvl == "EPOCH"]
        self.assertEqual(len(epoch_lines), 1)
        self.assertIn("D 3.0000", epoch_lines[0])  # (2+4)/2 的均值
        self.assertEqual(state["count"], 0, "报完均值必须清零计数")
        self.assertEqual(state["sum"]["d"], 0.0)

    def test_empty_epoch_does_not_divide_by_zero(self):
        log, on_epoch, _, _, _, _ = _reporter()
        on_epoch(epoch=1, total=10)  # 无 loss 上报，n=max(count,1) 兜底
        self.assertEqual(len([m for m, lvl in log.logged if lvl == "EPOCH"]), 1)


class TestTrainerLogRouting(unittest.TestCase):
    def test_epoch_lines_go_to_file_only(self):
        log, _, on_trainer_log, _, _, _ = _reporter()
        on_trainer_log("epoch 5 内部明细")
        on_trainer_log("普通训练日志")
        self.assertEqual(log.file_only_msgs, ["epoch 5 内部明细"])
        self.assertEqual([m for m, _ in log.logged], ["普通训练日志"])


if __name__ == "__main__":
    unittest.main()