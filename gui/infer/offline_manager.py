"""离线推理管理器 — 负责离线音频文件转换"""
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QFileDialog
import os

from rvc.inference import OfflineWorker

if TYPE_CHECKING:
    from gui.infer.window import MainWindow


class OfflineManager:
    """管理离线音频文件转换流程"""

    def __init__(self, window: 'MainWindow'):
        self.window = window
        self.worker = None

    def browse_file(self, target_widget, kind: str) -> None:
        """浏览文件选择"""
        if kind == "in":
            path, _ = QFileDialog.getOpenFileName(
                self.window, "选择输入文件", "", "音频 (*.wav *.mp3 *.flac)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self.window, "保存音频", "", "WAV (*.wav)"
            )

        if path:
            target_widget.setText(path)
            if kind == "in" and not self.window.off_out.text():
                base, _ = os.path.splitext(path)
                self.window.off_out.setText(base + "_converted.wav")

    def start_conversion(self) -> None:
        """开始离线转换"""
        inp = self.window.off_in.text().strip()
        out = self.window.off_out.text().strip()

        if not inp:
            self.window._show_warning("请选择输入文件")
            return
        if not os.path.exists(inp):
            self.window._show_warning(f"文件不存在: {inp}")
            return
        if not out:
            base, _ = os.path.splitext(inp)
            out = base + "_converted.wav"
            self.window.off_out.setText(out)

        if not self.window.model_manager.active_card:
            self.window._show_warning("请先在「模型」中选择一个模型")
            return

        pth = self.window.model_manager.active_card.pth_edit.text().strip()
        if not pth:
            self.window._show_warning("模型路径为空")
            return

        if self.window.engine.running:
            self.window._show_warning("请先停止实时变声")
            return

        # 启动转换
        from gui.infer.widgets import _sl_value_as_float
        card = self.window.model_manager.active_card
        idx = card.idx_edit.text().strip()
        pitch = card.pit_sl.value()
        f0method = self.window.f0_combo.currentText()

        self.worker = OfflineWorker(
            inp, out, pth, idx, pitch, f0method,
            _sl_value_as_float(card.ir_sl),
            _sl_value_as_float(card.rms_sl),
            (_sl_value_as_float(card.gen_sl) - 0.5) * 4,
            _sl_value_as_float(card.protect_sl),
            self.window.eq_en.isChecked(),
            _sl_value_as_float(self.window.eq_sub),
            _sl_value_as_float(self.window.eq_lo),
            _sl_value_as_float(self.window.eq_mi),
            _sl_value_as_float(self.window.eq_hi_mid),
            _sl_value_as_float(self.window.eq_hi),
            _sl_value_as_float(self.window.rev_sl),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.window.off_btn.setEnabled(False)
        self.window.off_btn.setText("转换中...")
        self.window.off_status.setText("初始化...")

    def _on_progress(self, current: int, total: int) -> None:
        """更新进度"""
        self.window.off_status.setText(f"进度: {current}/{total}")

    def _on_finished(self, path: str) -> None:
        """转换完成"""
        self.window.off_btn.setEnabled(True)
        self.window.off_btn.setText("开始转换")
        self.window.off_status.setText("完成")
        if self.worker:
            self.worker.wait()
            self.worker = None

    def _on_error(self, msg: str) -> None:
        """转换出错"""
        self.window.off_btn.setEnabled(True)
        self.window.off_btn.setText("开始转换")
        self.window.off_status.setText("错误")
        if self.worker:
            self.worker.wait()
            self.worker = None
        self.window._show_error(f"离线推理错误: {str(msg).strip().splitlines()[-1]}")

    def cleanup(self) -> None:
        """清理资源"""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(2000)
