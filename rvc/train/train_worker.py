import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from configs.config import Config
from rvc.train.extract_f0 import F0Extractor
from rvc.train.extract_feature import HuBERTExtractor
from rvc.train.preprocess import PreProcessor, generate_filelist, manifest_matches
from rvc.train.trainer import TrainConfig, Trainer


class TrainWorker(QThread):
    stage_changed = Signal(str)
    progress = Signal(int, int)
    log_message = Signal(str)
    loss_update = Signal(dict)
    epoch_done = Signal(int, int)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, options: dict, step: str = "all"):
        super().__init__()
        self.options = options
        self.step = step
        self._stop_requested = False
        self._trainer = None

    def request_stop(self):
        self._stop_requested = True
        if self._trainer is not None:
            self._trainer.stop()
        self.stage_changed.emit("正在停止")
        self.log_message.emit("收到停止请求，将在当前步骤结束后保存退出")

    def _check_stop(self):
        if self._stop_requested:
            raise RuntimeError("训练已停止")

    def run(self):
        try:
            self._run_impl()
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb.strip().splitlines()[-1])
            self.log_message.emit(tb)
            self.finished.emit(False, "训练失败")

    def _run_impl(self):
        config = Config()
        exp_dir = Path("logs") / self.options["exp_name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        sr = 48000 if self.options["sr"] == "48k" else 32000

        if self.step != "preprocess" and exp_dir.exists() and not manifest_matches(exp_dir, self.options["input_dir"], sr, 3.7):
            raise RuntimeError("实验目录与当前输入不一致，请先重新执行预处理")

        steps = {
            "preprocess": [self._step_preprocess],
            "f0": [self._step_f0],
            "feature": [self._step_feature],
            "train": [self._step_train],
            "all": [self._step_preprocess, self._step_f0, self._step_feature, self._step_train],
        }

        for step_fn in steps[self.step]:
            step_fn(config, exp_dir, sr)
            if self._stop_requested:
                self.finished.emit(False, "已停止")
                return

        self.stage_changed.emit("完成")
        self.finished.emit(True, "流程完成")

    def _step_preprocess(self, config, exp_dir, sr):
        self._check_stop()
        self.stage_changed.emit("预处理音频")
        self.log_message.emit("开始预处理音频")
        PreProcessor(self.options["input_dir"], str(exp_dir), sr).run(self.progress.emit)
        self._check_stop()
        self.log_message.emit("预处理完成")

    def _step_f0(self, config, exp_dir, sr):
        self._check_stop()
        self.stage_changed.emit("提取 F0")
        self.log_message.emit("开始提取 F0")
        extractor = F0Extractor(config.device, config.is_half)
        if self._stop_requested:
            extractor.request_stop()
        extractor.run(str(exp_dir), self.progress.emit)
        self._check_stop()
        self.log_message.emit("F0 提取完成")

    def _step_feature(self, config, exp_dir, sr):
        self._check_stop()
        self.stage_changed.emit("提取 HuBERT 特征")
        self.log_message.emit("开始提取 HuBERT 特征")
        extractor = HuBERTExtractor(config.device, config.is_half)
        if self._stop_requested:
            extractor.request_stop()
        extractor.run(str(exp_dir), self.progress.emit)
        self._check_stop()
        self.log_message.emit("HuBERT 特征提取完成")

    def _step_train(self, config, exp_dir, sr):
        self.stage_changed.emit("生成训练列表")
        filelist, count = generate_filelist(str(exp_dir))
        self.log_message.emit(f"训练样本数: {count}")
        if count == 0:
            raise RuntimeError("没有可训练样本")

        self.stage_changed.emit("训练模型")
        self.log_message.emit("开始训练模型")
        train_config = TrainConfig(
            exp_dir=str(exp_dir),
            sr=sr,
            epochs=self.options["epochs"],
            batch_size=self.options["batch_size"],
            save_every_epoch=self.options["save_every_epoch"],
            learning_rate=self.options["learning_rate"],
            pretrain_g=self.options.get("pretrain_g", ""),
            pretrain_d=self.options.get("pretrain_d", ""),
            fp16_run=config.is_half,
            device=config.device,
        )
        self._trainer = Trainer(train_config, self.epoch_done.emit, self.log_message.emit, self.loss_update.emit)
        output = self._trainer.train()
        self.log_message.emit(f"模型已导出: {output}")
