"""训练 GUI 主窗口"""
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QMessageBox, QWidget, QVBoxLayout, QTabWidget

from configs.config import load_state_json, save_state_json, state_path
from rvc.train.train_worker import TrainWorker
from gui.train.widgets import ToolThread
from gui.train.tabs.settings_tab import build_settings_tab
from gui.train.tabs.train_tab import build_train_tab
from gui.train.tabs.tools_tab import build_tools_tab

TRAIN_STATE_KEY = "train"


class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 训练")
        self.resize(360, 253)  # 540/1.5=360, 380/1.5≈253
        self.worker = None
        self._tool_thread = None
        self._build_ui()
        self._load_cfg()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(build_settings_tab(self), "设置")
        tabs.addTab(build_train_tab(self), "训练")
        tabs.addTab(build_tools_tab(self), "工具")
        layout.addWidget(tabs)

        self.setCentralWidget(central)

    # ── 训练逻辑 ────────────────────────────────────────────

    def _on_sr_changed(self, text: str):
        sr = "48k" if text == "48k" else "32k"
        if not self.pretrain_g.text().strip():
            path = Path(f"assets/pretrained_v2/f0G{sr}.pth")
            if path.exists():
                self.pretrain_g.setText(str(path))
        if not self.pretrain_d.text().strip():
            path = Path(f"assets/pretrained_v2/f0D{sr}.pth")
            if path.exists():
                self.pretrain_d.setText(str(path))

    def _start_step(self, step: str):
        try:
            options = self._collect_options()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self._set_running(True)
        self.log_edit.clear()
        self.worker = TrainWorker(options, step)
        self.worker.stage_changed.connect(self.on_stage_changed)
        self.worker.progress.connect(self.on_progress)
        self.worker.log_message.connect(self.on_log)
        self.worker.loss_update.connect(self.on_loss)
        self.worker.epoch_done.connect(self.on_epoch)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "训练错误", msg))
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("停止中...")
            self.stage_label.setText("当前阶段: 正在请求停止")

    def _collect_options(self):
        exp_name = self.exp_name.text().strip()
        input_dir = self.input_dir.text().strip()
        if not exp_name:
            raise ValueError("实验名不能为空")
        if not input_dir or not Path(input_dir).exists():
            raise ValueError("请选择有效的音频目录")
        try:
            lr = float(self.learning_rate.text().strip())
        except ValueError as exc:
            raise ValueError("学习率格式不正确") from exc
        return {
            "exp_name": exp_name,
            "input_dir": input_dir,
            "sr": self.sample_rate.currentText(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "save_every_epoch": self.save_every.value(),
            "learning_rate": lr,
            "pretrain_g": self.pretrain_g.text().strip(),
            "pretrain_d": self.pretrain_d.text().strip(),
        }

    def _set_running(self, running: bool):
        for btn in [self.btn_preprocess, self.btn_f0, self.btn_feature, self.btn_train, self.btn_all]:
            btn.setEnabled(not running)
            btn.setStyleSheet("QPushButton{color:#6c757d}" if running else "")
        self.stop_btn.setEnabled(running)
        self.stop_btn.setText("停止训练" if running else "停止训练")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#dc3545;color:white;border:none;padding:4px 8px;border-radius:3px}QPushButton:hover{background:#c82333}"
            if running else ""
        )
        self.stage_label.setStyleSheet("color:#3b82f6;font-weight:bold" if running else "")
        for widget in [self.exp_name, self.input_dir, self.sample_rate, self.epochs, self.batch_size, self.save_every, self.learning_rate, self.pretrain_g, self.pretrain_d]:
            widget.setEnabled(not running)

    # ── 配置持久化 ──────────────────────────────────────────

    def _save_cfg(self):
        cfg = {
            "exp_name": self.exp_name.text().strip(),
            "input_dir": self.input_dir.text().strip(),
            "sr": self.sample_rate.currentText(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "save_every": self.save_every.value(),
            "learning_rate": self.learning_rate.text().strip(),
            "pretrain_g": self.pretrain_g.text().strip(),
            "pretrain_d": self.pretrain_d.text().strip(),
        }
        save_state_json(TRAIN_STATE_KEY, cfg)

    def _load_cfg(self):
        cfg = load_state_json(TRAIN_STATE_KEY, {})
        if cfg.get("exp_name"):
            self.exp_name.setText(cfg["exp_name"])
        if cfg.get("input_dir"):
            self.input_dir.setText(cfg["input_dir"])
        if cfg.get("sr"):
            idx = self.sample_rate.findText(cfg["sr"])
            if idx >= 0:
                self.sample_rate.setCurrentIndex(idx)
        if cfg.get("epochs"):
            self.epochs.setValue(cfg["epochs"])
        if cfg.get("batch_size"):
            self.batch_size.setValue(cfg["batch_size"])
        if cfg.get("save_every"):
            self.save_every.setValue(cfg["save_every"])
        if cfg.get("learning_rate"):
            self.learning_rate.setText(cfg["learning_rate"])
        if cfg.get("pretrain_g"):
            self.pretrain_g.setText(cfg["pretrain_g"])
        if cfg.get("pretrain_d"):
            self.pretrain_d.setText(cfg["pretrain_d"])

    # ── 训练回调 ────────────────────────────────────────────

    def on_stage_changed(self, stage: str):
        self.stage_label.setText(f"当前阶段: {stage}")
        self.stage_label.setStyleSheet("color:#3b82f6;font-weight:bold")
        self.progress_bar.setValue(0)

    def on_progress(self, current: int, total: int):
        self.progress_bar.setValue(int(current * 100 / max(total, 1)))

    def on_epoch(self, epoch: int, total: int):
        self.epoch_label.setText(f"Epoch: {epoch} / {total}")
        self.on_progress(epoch, total)

    def on_loss(self, data: dict):
        self.loss_label.setText(
            "Loss: "
            f"D {data['loss_d']:.4f} | G {data['loss_g']:.4f} | "
            f"Mel {data['loss_mel']:.4f} | KL {data['loss_kl']:.4f} | FM {data['loss_fm']:.4f}"
        )

    def on_log(self, message: str):
        self.log_edit.append(message.rstrip())
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def on_finished(self, success: bool, message: str):
        self._set_running(False)
        self.stop_btn.setText("停止训练")
        self.on_log(message)
        if success:
            self.stage_label.setStyleSheet("color:#28a745;font-weight:bold")
            QMessageBox.information(self, "完成", message)
        else:
            self.stage_label.setStyleSheet("color:#dc3545;font-weight:bold")

    def closeEvent(self, event):
        try:
            self._save_cfg()
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2000)
        if self._tool_thread and self._tool_thread.isRunning():
            self._tool_thread.wait(5000)
        event.accept()
