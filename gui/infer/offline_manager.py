"""离线推理管理器 — 负责离线音频文件转换"""
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QFileDialog
import os

from gui.infer.workers import OfflineWorker
from rvc.inference.offline_config import OfflineConfig
from gui.infer.utils import format_error_message
from gui.infer.widgets import _sl_value_as_float

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
                self.window, "选择输入文件", "", "音频 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self.window, "保存音频", "", "WAV (*.wav)"
            )

        if path:
            target_widget.setText(path)
            if kind == "in" and not self.window.offline_output.text():
                base, _ = os.path.splitext(path)
                self.window.offline_output.setText(base + "_converted.wav")

    def start_conversion(self) -> None:
        """开始离线转换"""
        inp = self.window.offline_input.text().strip()
        out = self.window.offline_output.text().strip()

        if not inp:
            self.window._show_warning("请选择输入文件")
            return
        if not os.path.exists(inp):
            self.window._show_warning(f"文件不存在: {inp}")
            return
        if not out:
            base, _ = os.path.splitext(inp)
            out = base + "_converted.wav"
            self.window.offline_output.setText(out)

        if not self.window.model_manager.active_card:
            self.window._show_warning("请先在「模型」中选择一个模型")
            return

        card = self.window.model_manager.active_card
        pth = card.pth_edit.text().strip()
        if not pth:
            self.window._show_warning("模型路径为空")
            return

        if self.window.engine.running:
            self.window._show_warning("请先停止实时变声")
            return

        # 构建配置并启动转换
        config = OfflineConfig(
            input_path=self.window.offline_input.text().strip(),
            output_path=self.window.offline_output.text().strip(),
            model_path=card.pth_edit.text().strip(),
            index_path=card.idx_edit.text().strip(),
            pitch=card.pitch_slider.value(),
            f0method=self.window.f0_combo.currentText(),
            index_rate=_sl_value_as_float(card.index_rate_slider),
            rms_mix=_sl_value_as_float(card.rms_mix_slider),
            protect=_sl_value_as_float(card.protect_slider),
            eq_enabled=self.window.eq_enable_checkbox.isChecked(),
            eq_bands={
                'sub': _sl_value_as_float(self.window.eq_sub_slider),
                'low': _sl_value_as_float(self.window.eq_low_slider),
                'mid': _sl_value_as_float(self.window.eq_mid_slider),
                'hi_mid': _sl_value_as_float(self.window.eq_hi_mid_slider),
                'high': _sl_value_as_float(self.window.eq_high_slider),
            },
            reverb_mix=_sl_value_as_float(self.window.reverb_slider),
        )
        self.worker = OfflineWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.window.offline_button.setEnabled(False)
        self.window.offline_button.setText("转换中...")
        self.window.offline_status.setText("初始化...")
        self.window.offline_progress.setValue(0)

    def _on_progress(self, current: int, total: int) -> None:
        """更新进度"""
        self.window.offline_status.setText(f"进度: {current}/{total}")
        if total > 0:
            self.window.offline_progress.setValue(current)

    def _on_finished(self, path: str) -> None:
        """转换完成"""
        self.window.offline_button.setEnabled(True)
        self.window.offline_button.setText("开始转换")
        self.window.offline_status.setText("完成")
        if self.worker:
            self.worker.wait()
            self.worker = None

    def _on_error(self, msg: str) -> None:
        """转换出错"""
        self.window.offline_button.setEnabled(True)
        self.window.offline_button.setText("开始转换")
        self.window.offline_status.setText("错误")
        if self.worker:
            self.worker.wait()
            self.worker = None
        self.window._show_error(f"离线推理错误: {format_error_message(msg)}")

    def cleanup(self) -> None:
        """清理资源"""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(2000)
