"""离线推理管理器 — 负责离线音频文件转换"""
import logging
import os
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog

from rvc.core.config import OfflineConfig
from gui.infer.view.widgets import _sl_value_as_float
from gui.infer.viewmodel.param_binding import collect_gui_state, format_error_message, gender_to_formant

if TYPE_CHECKING:
    from gui.infer.view.main_window import MainWindow


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

        # 构建配置并启动转换（效果/音高参数统一从 GUI 状态读取，与实时一致）
        # 离线推理使用独立的 RealtimeEngine 实例，可与实时变声同时运行（需显存足够）
        try:
            state = collect_gui_state(self.window)
            inf = state.inference
            config = OfflineConfig(
                input_path=self.window.offline_input.text().strip(),
                output_path=self.window.offline_output.text().strip(),
                model_path=card.pth_edit.text().strip(),
                pitch=inf.pitch,
                formant=inf.formant,
                protect=inf.protect,
                f0_method=inf.f0_method,
                rms_mix=inf.rms_mix,
                break_protect=inf.break_protect,
                hubert=card.hubert_combo.currentText(),
            )
        except Exception as exc:
            import traceback
            logger.error("离线转换初始化失败:\n%s", traceback.format_exc())
            self.window._show_error(f"离线转换初始化失败: {exc}")
            return

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
        # 与 _on_finished 保持一致：不 deleteLater / 不置 None——自定义 finished/error
        # 信号在 run() 内发出，此刻线程可能尚未真正退出，提前析构有竞态。
        # 引用保留到下次 start_conversion 时被新 worker 覆盖，一次只多占一个对象。
        self.window._show_error(f"离线推理错误: {format_error_message(msg)}")


logger = logging.getLogger(__name__)


class OfflineWorker(QThread):
    """离线推理线程：模拟播放→转换→写录，复用实时引擎全链路（显存封顶）。"""
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, cfg: OfflineConfig):
        super().__init__()
        self.cfg = cfg

    def run(self):
        import torch  # 惰性导入，避免 GUI 启动时加载 torch

        try:
            self._do_run()
        except Exception:
            tb = traceback.format_exc()
            logger.error("离线推理失败:\n%s", tb)
            self.error.emit(tb.strip())
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _do_run(self):
        from rvc.audio.realtime_engine import RealtimeEngine

        self.progress.emit(0, 100)
        engine = RealtimeEngine(self.cfg)
        engine.load_model(self.cfg.model_path, hubert=self.cfg.hubert)
        self.progress.emit(20, 100)

        def _progress(cur, total):
            # 推理进度占 80%（模型加载占前 20%），避免进度回退
            pct = 20 + int(cur * 80 / total) if total > 0 else 20
            self.progress.emit(pct, 100)

        engine.process_file(self.cfg, progress_cb=_progress)
        self.progress.emit(100, 100)
        self.finished.emit(self.cfg.output_path)

