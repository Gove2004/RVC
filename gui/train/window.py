"""训练 GUI 主窗口"""
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QMessageBox, QWidget, QVBoxLayout, QTabWidget, QFileDialog,
)

from gui.configs import TrainGuiState, load_config, save_config
from gui.train.workers import TrainWorker, VocalExtractWorker
from gui.train.widgets import ToolThread
from gui.train.tabs.settings_tab import build_settings_tab
from gui.train.tabs.train_tab import build_train_tab
from gui.train.tabs.tools_tab import build_tools_tab
from gui.train.tabs.vocal_tab import build_vocal_tab
from gui.styles import ButtonStyles, LabelStyles, Layout
from rvc.runtime.paths import PRETRAINED_ROOT


class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 训练")
        self.resize(470, 400)
        self.worker = None
        self._sep_worker = None
        self._tool_thread = None
        self._last_loss_text = ""
        self._build_ui()
        self._load_cfg()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(build_settings_tab(self), "设置")
        tabs.addTab(build_train_tab(self), "训练")
        tabs.addTab(build_vocal_tab(self), "人声提纯")
        tabs.addTab(build_tools_tab(self), "工具")
        layout.addWidget(tabs)

        self.setCentralWidget(central)

    # ── 训练逻辑 ────────────────────────────────────────────

    def _on_sr_changed(self, text: str):
        """采样率改变时自动匹配预训练 G/D（存在即填，不存在清空；用户可再手动改）。

        选择 40k/48k 不再需要手动指定 G/D——自动指向 assets/pretrained/f0G/f0D{sr}k.pth。
        G/D 只是收敛加速的可选底座：留空 = 从零训练（收敛慢、音质起步低）。
        """
        for widget, key in ((self.pretrain_g, "G"), (self.pretrain_d, "D")):
            path = PRETRAINED_ROOT / f"f0{key}{text}.pth"
            widget.setText(str(path) if path.exists() else "")

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
        self.worker.batch_done.connect(self.on_batch)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "训练错误", msg))
        self.worker.finished.connect(self.on_finished)
        self._batch_t0 = None  # 当前 epoch 首 batch 时间戳（计算 it/s）
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("停止中...")
            self.stage_label.setText("当前阶段: 正在请求停止")

    # ── 人声提纯 ──────────────────────────────────────────

    def _sep_browse(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            line_edit.setText(path)

    def _sep_collect_options(self):
        input_dir = self.sep_input.text().strip()
        output_dir = self.sep_output.text().strip()
        if not input_dir or not Path(input_dir).exists():
            raise ValueError("请选择有效的输入文件夹")
        if not output_dir:
            raise ValueError("请选择输出文件夹")
        if Path(input_dir).resolve() == Path(output_dir).resolve():
            raise ValueError("输出文件夹不能和输入文件夹相同（会覆盖原始素材）")
        options = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "model": self.sep_model.currentData(),
            "out_sr": self.sep_out_sr.currentText(),
        }
        from rvc.tools.separate import POST_KEYS

        for key in POST_KEYS:
            options[f"do_{key}"] = getattr(self, f"sep_do_{key}").isChecked()
        return options

    def _sep_start(self):
        try:
            options = self._sep_collect_options()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self._sep_set_running(True)
        self.sep_log.clear()
        self.sep_progress.setValue(0)
        self._sep_worker = VocalExtractWorker(options)
        self._sep_worker.stage_changed.connect(self._sep_on_stage)
        self._sep_worker.progress.connect(self._sep_on_progress)
        self._sep_worker.log_message.connect(self._sep_on_log)
        self._sep_worker.error.connect(lambda msg: QMessageBox.critical(self, "提纯失败", msg))
        self._sep_worker.cancelled.connect(self._sep_on_cancelled)
        self._sep_worker.finished.connect(self._sep_on_finished)
        self._sep_worker.start()

    def _sep_stop(self):
        if self._sep_worker:
            self._sep_worker.request_stop()
            self.sep_stop.setEnabled(False)
            self.sep_stop.setText("停止中...")

    def _sep_set_running(self, running: bool):
        self.sep_start.setEnabled(not running)
        self.sep_stop.setEnabled(running)
        self.sep_stop.setText("停止")
        for widget in [self.sep_input, self.sep_output, self.sep_model, self.sep_out_sr]:
            widget.setEnabled(not running)
        from rvc.tools.separate import POST_KEYS

        for key in POST_KEYS:
            getattr(self, f"sep_do_{key}").setEnabled(not running)

    def _sep_on_stage(self, stage: str):
        self.sep_status.setText(stage)

    def _sep_on_progress(self, current: int, total: int):
        self.sep_progress.setValue(int(current * 100 / max(total, 1)))

    def _sep_on_log(self, message: str):
        self.sep_log.append(message.rstrip())
        self.sep_log.verticalScrollBar().setValue(self.sep_log.verticalScrollBar().maximum())

    def _sep_on_finished(self, success: bool, message: str):
        self._sep_set_running(False)
        self._sep_on_log(message)
        if success:
            self.sep_status.setStyleSheet(LabelStyles.status("success"))
            QMessageBox.information(self, "完成", message)
        else:
            self.sep_status.setStyleSheet(LabelStyles.status("error"))

    def _sep_on_cancelled(self, message: str):
        self._sep_set_running(False)
        self._sep_on_log(message)
        self.sep_status.setStyleSheet(LabelStyles.status("warning"))

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
            "hubert": self.hubert.currentText(),
        }

    def _set_running(self, running: bool):
        for btn in [self.btn_preprocess, self.btn_f0, self.btn_feature, self.btn_train, self.btn_all]:
            btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.stop_btn.setStyleSheet(ButtonStyles.danger())
            self.stage_label.setStyleSheet(LabelStyles.status("info"))
        else:
            self.stop_btn.setStyleSheet(ButtonStyles.muted())
            self.stage_label.setStyleSheet("")
        for widget in [self.exp_name, self.input_dir, self.sample_rate, self.hubert, self.epochs, self.batch_size, self.save_every, self.learning_rate, self.pretrain_g, self.pretrain_d]:
            widget.setEnabled(not running)

    # ── 配置持久化 ──────────────────────────────────────────

    def collect_gui_state(self) -> TrainGuiState:
        return TrainGuiState(
            exp_name=self.exp_name.text().strip(),
            input_dir=self.input_dir.text().strip(),
            sample_rate=self.sample_rate.currentText(),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            save_every=self.save_every.value(),
            learning_rate=self.learning_rate.text().strip(),
            pretrain_g=self.pretrain_g.text().strip(),
            pretrain_d=self.pretrain_d.text().strip(),
            hubert=self.hubert.currentText(),
            sep_input_dir=self.sep_input.text().strip(),
            sep_output_dir=self.sep_output.text().strip(),
            sep_model=self.sep_model.currentData() or "htdemucs",
            sep_out_sr=self.sep_out_sr.currentText(),
            sep_dereverb=self.sep_do_dereverb.isChecked(),
            sep_karaoke=self.sep_do_karaoke.isChecked(),
            sep_denoise=self.sep_do_denoise.isChecked(),
        )

    def apply_gui_state(self, state: TrainGuiState) -> None:
        if state.exp_name:
            self.exp_name.setText(state.exp_name)
        if state.input_dir:
            self.input_dir.setText(state.input_dir)
        if state.sample_rate:
            idx = self.sample_rate.findText(state.sample_rate)
            if idx >= 0:
                self.sample_rate.setCurrentIndex(idx)
        self.epochs.setValue(state.epochs)
        self.batch_size.setValue(state.batch_size)
        self.save_every.setValue(state.save_every)
        if state.learning_rate:
            self.learning_rate.setText(state.learning_rate)
        if state.pretrain_g:
            self.pretrain_g.setText(state.pretrain_g)
        if state.pretrain_d:
            self.pretrain_d.setText(state.pretrain_d)
        if state.hubert:
            idx = self.hubert.findText(state.hubert)
            if idx >= 0:
                self.hubert.setCurrentIndex(idx)
        # 人声提纯 Tab
        if state.sep_input_dir:
            self.sep_input.setText(state.sep_input_dir)
        if state.sep_output_dir:
            self.sep_output.setText(state.sep_output_dir)
        idx = self.sep_model.findData(state.sep_model)
        if idx >= 0:
            self.sep_model.setCurrentIndex(idx)
        idx = self.sep_out_sr.findText(state.sep_out_sr)
        if idx >= 0:
            self.sep_out_sr.setCurrentIndex(idx)
        self.sep_do_dereverb.setChecked(state.sep_dereverb)
        self.sep_do_karaoke.setChecked(state.sep_karaoke)
        self.sep_do_denoise.setChecked(state.sep_denoise)

    def _save_cfg(self):
        config = load_config()
        config["train"] = self.collect_gui_state().to_dict()
        save_config(config)

    def _load_cfg(self):
        config = load_config()
        self.apply_gui_state(TrainGuiState.from_dict(config.get("train", {})))

    # ── 训练回调 ────────────────────────────────────────────

    def on_stage_changed(self, stage: str):
        self.stage_label.setText(f"当前阶段: {stage}")
        self.stage_label.setStyleSheet(LabelStyles.status("info"))
        self.progress_bar.setValue(0)
        self._batch_t0 = None

    def on_progress(self, current: int, total: int):
        self.progress_bar.setValue(int(current * 100 / max(total, 1)))

    def on_epoch(self, epoch: int, total: int):
        self.epoch_label.setText(f"Epoch: {epoch} / {total}")
        self.on_progress(epoch, total)

    def on_batch(self, epoch: int, batch: int, total: int):
        """batch 级进度：更新进度条 + 计算 it/s 与剩余时间"""
        now = time.monotonic()
        if batch == 1 or self._batch_t0 is None:
            self._batch_t0 = now
            it_s = 0.0
        else:
            elapsed = now - self._batch_t0
            it_s = (batch - 1) / max(elapsed, 1e-6)

        self.on_progress(batch, total)

        eta_text = ""
        if it_s > 0:
            remaining_batches = (total - batch) + max(self.worker.options["epochs"] - epoch, 0) * total
            eta_sec = remaining_batches / it_s
            eta_text = f" · ETA {int(eta_sec // 3600)}h{int(eta_sec % 3600 // 60)}m"
        speed = f"{it_s:.1f} it/s" if it_s > 0 else "-"
        self.epoch_label.setText(f"Epoch: {epoch} / {self.worker.options['epochs']} · Batch {batch}/{total} · {speed}{eta_text}")

    def on_loss(self, data: dict):
        # 每 batch 都会收到信号，这里做 150ms 节流防止高频刷新 UI
        now = time.monotonic()
        if now - getattr(self, "_last_loss_t", 0.0) < 0.15:
            return
        self._last_loss_t = now
        text = (
            "Loss: "
            f"D {data['loss_d']:.4f} | G {data['loss_g']:.4f} | "
            f"Mel {data['loss_mel']:.4f} | KL {data['loss_kl']:.4f} | FM {data['loss_fm']:.4f}"
        )
        if text != self._last_loss_text:
            self.loss_label.setText(text)
            self._last_loss_text = text

    def on_log(self, message: str):
        self.log_edit.append(message.rstrip())
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def on_finished(self, success: bool, message: str):
        self._set_running(False)
        self.stop_btn.setText("停止训练")
        self._batch_t0 = None
        self.on_log(message)
        if success:
            self.stage_label.setStyleSheet(LabelStyles.status("success"))
            QMessageBox.information(self, "完成", message)
        else:
            self.stage_label.setStyleSheet(LabelStyles.status("error"))

    def closeEvent(self, event):
        try:
            self._save_cfg()
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            if not self.worker.wait(2000):
                # 停止超时：若直接销毁窗口，QThread 对象会被销毁而线程仍在运行
                # （"Destroyed while thread is still running"）。改为阻止关闭并提示。
                event.ignore()
                QMessageBox.information(self, "提示", "任务正在停止，请稍候再关闭窗口")
                return
        if self._tool_thread and self._tool_thread.isRunning():
            if not self._tool_thread.wait(5000):
                event.ignore()
                QMessageBox.information(self, "提示", "工具任务正在停止，请稍候再关闭窗口")
                return
        if self._sep_worker and self._sep_worker.isRunning():
            # 提纯的取消点只在文件之间，单个长音频可能要跑很久才响应
            self._sep_worker.request_stop()
            if not self._sep_worker.wait(5000):
                event.ignore()
                QMessageBox.information(self, "提示", "人声提纯正在停止，请稍候再关闭窗口")
                return
        event.accept()
