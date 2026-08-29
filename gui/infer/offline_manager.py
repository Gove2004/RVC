"""离线推理管理器 — 负责离线音频文件转换"""
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QFileDialog
import os

from gui.infer.workers import OfflineWorker
from rvc.inference.offline_config import OfflineConfig
from gui.infer.utils import format_error_message
from gui.infer.widgets import _sl_value_as_float
from gui.infer.param_binding import collect_gui_state, gender_to_formant

if TYPE_CHECKING:
    from gui.infer.window import MainWindow


class OfflineManager:
    """管理离线音频文件转换流程"""

    def __init__(self, window: 'MainWindow'):
        self.window = window
        self.worker = None
        self._converting = False

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
        if self._converting:
            self.window._show_warning("已有转换任务正在运行")
            return
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

        # 构建配置并启动转换（效果/音高参数统一从 GUI 状态读取，与实时一致）
        state = collect_gui_state(self.window)
        config = OfflineConfig(
            input_path=self.window.offline_input.text().strip(),
            output_path=self.window.offline_output.text().strip(),
            model_path=card.pth_edit.text().strip(),
            index_path=card.idx_edit.text().strip(),
            pitch=card.pitch_slider.value(),
            f0method=state.f0method,
            index_rate=_sl_value_as_float(card.index_rate_slider),
            rms_mix=state.rms_mix,
            protect=state.protect,
            gender=gender_to_formant(_sl_value_as_float(card.gender_slider)),  # 与实时同一换算
            break_enable=state.break_enable,
            break_src_hz=state.break_src_hz,
        )
        self.worker = OfflineWorker(config)
        self._converting = True
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
        """转换完成 - worker 会自动退出并自行删除，不阻塞等待"""
        self._converting = False
        self.window.offline_button.setEnabled(True)
        self.window.offline_button.setText("开始转换")
        self.window.offline_status.setText("完成")
        # QThread 的 run() 返回后线程自动结束，worker 对象稍后由 Qt 事件循环删除
        # 不需要 deleteLater() / wait()，避免了阻塞和潜在的"Destroyer while thread still running"警告

    def _on_error(self, msg: str) -> None:
        """转换出错"""
        self._converting = False
        self.window.offline_button.setEnabled(True)
        self.window.offline_button.setText("开始转换")
        self.window.offline_status.setText("错误")
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.window._show_error(f"离线推理错误: {format_error_message(msg)}")
