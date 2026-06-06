import io
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rvc.train.train_worker import TrainWorker

TRAIN_CONFIG_PATH = Path("configs/inuse/train_config.json")


def _browse_directory(parent, line_edit: QLineEdit):
    path = QFileDialog.getExistingDirectory(parent, "选择音频目录")
    if path:
        line_edit.setText(path)


def _browse_file(parent, line_edit: QLineEdit):
    path, _ = QFileDialog.getOpenFileName(parent, "选择模型文件", "", "PyTorch (*.pth)")
    if path:
        line_edit.setText(path)


class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 训练")
        self.resize(820, 780)
        self.worker = None
        self._build_ui()
        self._load_cfg()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_train_group())
        layout.addWidget(self._build_progress_group())
        layout.addLayout(self._build_buttons())

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group, 1)

        self.setCentralWidget(central)

    def _build_data_group(self):
        group = QGroupBox("数据设置")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        self.exp_name = QLineEdit(f"rvc_{time.strftime('%Y%m%d_%H%M%S')}")
        self.input_dir = QLineEdit()
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda: _browse_directory(self, self.input_dir))
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["48k", "32k"])
        self.sample_rate.currentTextChanged.connect(self._on_sr_changed)

        grid.addWidget(QLabel("实验名"), 0, 0)
        grid.addWidget(self.exp_name, 0, 1, 1, 2)
        grid.addWidget(QLabel("音频目录"), 1, 0)
        grid.addWidget(self.input_dir, 1, 1)
        grid.addWidget(browse, 1, 2)
        grid.addWidget(QLabel("采样率"), 2, 0)
        grid.addWidget(self.sample_rate, 2, 1, 1, 2)
        return group

    def _build_train_group(self):
        group = QGroupBox("训练参数")
        form = QFormLayout(group)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(6)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(2000)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(4)
        self.save_every = QSpinBox()
        self.save_every.setRange(1, 100000)
        self.save_every.setValue(200)
        self.learning_rate = QLineEdit("1e-4")

        self.pretrain_g = QLineEdit()
        self.pretrain_d = QLineEdit()
        g_row = QHBoxLayout()
        g_btn = QPushButton("浏览")
        g_btn.clicked.connect(lambda: _browse_file(self, self.pretrain_g))
        g_row.addWidget(self.pretrain_g)
        g_row.addWidget(g_btn)
        d_row = QHBoxLayout()
        d_btn = QPushButton("浏览")
        d_btn.clicked.connect(lambda: _browse_file(self, self.pretrain_d))
        d_row.addWidget(self.pretrain_d)
        d_row.addWidget(d_btn)

        form.addRow("Epoch", self.epochs)
        form.addRow("Batch size", self.batch_size)
        form.addRow("保存频率", self.save_every)
        form.addRow("学习率", self.learning_rate)
        form.addRow("预训练 G", g_row)
        form.addRow("预训练 D", d_row)
        return group

    def _build_progress_group(self):
        group = QGroupBox("训练进度")
        layout = QVBoxLayout(group)
        self.stage_label = QLabel("当前阶段: 未开始")
        self.epoch_label = QLabel("Epoch: -")
        self.loss_label = QLabel("Loss: -")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.epoch_label)
        layout.addWidget(self.loss_label)
        layout.addWidget(self.progress_bar)
        return group

    def _build_buttons(self):
        layout = QHBoxLayout()
        layout.addStretch(1)

        self.btn_preprocess = QPushButton("1. 预处理")
        self.btn_f0 = QPushButton("2. 提取F0")
        self.btn_feature = QPushButton("3. 提取特征")
        self.btn_train = QPushButton("4. 训练")
        self.btn_all = QPushButton("一键全流程")
        self.stop_btn = QPushButton("停止训练")

        self.btn_preprocess.clicked.connect(lambda: self._start_step("preprocess"))
        self.btn_f0.clicked.connect(lambda: self._start_step("f0"))
        self.btn_feature.clicked.connect(lambda: self._start_step("feature"))
        self.btn_train.clicked.connect(lambda: self._start_step("train"))
        self.btn_all.clicked.connect(lambda: self._start_step("all"))
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.btn_preprocess)
        layout.addWidget(self.btn_f0)
        layout.addWidget(self.btn_feature)
        layout.addWidget(self.btn_train)
        layout.addWidget(self.btn_all)
        layout.addWidget(self.stop_btn)
        return layout

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
        self.stop_btn.setEnabled(running)
        for widget in [self.exp_name, self.input_dir, self.sample_rate, self.epochs, self.batch_size, self.save_every, self.learning_rate, self.pretrain_g, self.pretrain_d]:
            widget.setEnabled(not running)

    def _save_cfg(self):
        TRAIN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        TRAIN_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cfg(self):
        if not TRAIN_CONFIG_PATH.exists():
            return
        try:
            cfg = json.loads(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
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

    def on_stage_changed(self, stage: str):
        self.stage_label.setText(f"当前阶段: {stage}")
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
        self.on_log(message)
        if success:
            QMessageBox.information(self, "完成", message)

    def closeEvent(self, event):
        try:
            self._save_cfg()
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    app = QApplication(sys.argv)
    win = TrainWindow()
    win.show()
    sys.exit(app.exec())
