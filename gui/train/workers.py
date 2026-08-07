"""训练 GUI 线程 worker。"""
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from rvc.runtime import Config
from rvc.train.extract_f0 import F0Extractor
from rvc.train.extract_feature import HuBERTExtractor
from rvc.train.preprocess import PreProcessor, generate_filelist, manifest_diff_reason
from rvc.train.trainer import TrainConfig, Trainer


class TrainWorker(QThread):
    stage_changed = Signal(str)
    progress = Signal(int, int)
    log_message = Signal(str)
    loss_update = Signal(dict)
    epoch_done = Signal(int, int)
    batch_done = Signal(int, int, int)  # epoch, batch, total_batches
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, options: dict, step: str = "all"):
        super().__init__()
        self.options = options
        self.step = step
        self._stop_requested = False
        self._trainer = None

    def request_stop(self):
        # 只置停止标志并通知 trainer；不做 cleanup/置 None——
        # trainer 在 worker 线程中运行，主线程提前清理会导致
        # _train_epoch 访问 None 模型而崩溃。清理由 _step_train 的
        # finally 在 train() 结束后于 worker 线程内执行。
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
            if self._stop_requested:
                # 用户主动停止：正常收尾，不弹错误窗
                self.finished.emit(False, "已停止")
                return
            tb = traceback.format_exc()
            self.error.emit(tb.strip().splitlines()[-1])
            self.log_message.emit(tb)
            self.finished.emit(False, "训练失败")

    def _run_impl(self):
        config = Config()
        exp_dir = Path("logs") / self.options["exp_name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        # 采样率来自 GUI（"40k"/"48k"）
        sr_text = str(self.options.get("sr", "48k")).strip().lower()
        sr = int(float(sr_text[:-1]) * 1000) if sr_text.endswith("k") else int(float(sr_text))

        # 仅对「不含预处理」的单独步骤做一致性检查；一键全流程(all)第一步
        # 就是预处理，_prepare_exp_dir 内部会按需清理旧数据重建，无需提前拦截
        if self.step not in ("preprocess", "all") and exp_dir.exists():
            reason = manifest_diff_reason(exp_dir, self.options["input_dir"], sr, 3.7)
            if reason:
                raise RuntimeError(f"实验目录与当前输入不一致：{reason}。请先重新执行预处理")

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

        # Validate pretrain paths before starting training
        for name in ("pretrain_g", "pretrain_d"):
            path = self.options.get(name, "")
            if path and not Path(path).exists():
                raise RuntimeError(f"预训练模型不存在: {path}")

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
            early_stop_patience=self.options.get("early_stop_patience", 0),
        )
        self._trainer = Trainer(train_config, self.epoch_done.emit, self.log_message.emit, self.loss_update.emit, self.batch_done.emit)
        try:
            output = self._trainer.train()
        finally:
            # 无论正常完成/停止/异常，都在 worker 线程内回收显存，避免与主线程竞态
            self._trainer.cleanup()
            self._trainer = None
        self.log_message.emit(f"模型已导出: {output}")
