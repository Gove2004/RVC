"""离线推理管理器 — 负责离线音频文件转换。"""

import logging
import os
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

from gui.state.bindings import collect_params
from rvc.core.config import OfflineParams
from rvc.core.errors import CancelledError

if TYPE_CHECKING:
    from gui.view.window import MainWindow

logger = logging.getLogger(__name__)

# 输出文件后缀：输出恒与输入同目录，只在文件名后追加该后缀（界面不可修改）
OUTPUT_SUFFIX = "_converted.wav"


def default_output_path(input_path: str) -> str:
    """由输入路径派生输出路径：同目录 + `<原名>_converted.wav`。"""
    if not input_path:
        return ""
    base, _ = os.path.splitext(input_path)
    return base + OUTPUT_SUFFIX


class OfflineConversion:
    """管理离线音频文件转换流程。"""

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.worker = None
        self._converting = False

    @property
    def is_converting(self) -> bool:
        """是否有离线转换在跑（供实时侧做互斥守卫）。"""
        return self._converting

    def start_conversion(self) -> None:
        """开始离线转换。"""
        if self._converting:
            self.window._show_warning("已有转换任务正在运行")
            return

        # 实时推理运行时禁止离线转换（共享模型缓存，同时运行会互相干扰 CUDA Graph）。
        # 经 controller.is_running 判断：引擎未构造时短路返回 False，不会触发惰性加载。
        if self.window.controller.is_running:
            self.window._show_warning("实时推理正在运行，请先停止实时推理再进行离线转换")
            return

        # offline_input_btn 在 offline_tab 构建时无条件创建（缺失即 AttributeError）
        inp = self.window.offline_input_btn.path().strip()

        if not inp:
            self.window._show_warning("请选择输入文件")
            return
        if not os.path.exists(inp):
            self.window._show_warning(f"文件不存在: {inp}")
            return

        # 输出路径由输入派生（界面上是只读文本，用户不可修改）
        out = default_output_path(inp)

        # model_path_btn 在 timbre_tab 构建时无条件创建（缺失即 AttributeError）
        pth = self.window.model_path_btn.path().strip()
        if not pth:
            self.window._show_warning("请先在「音色」中选择模型文件")
            return

        # 构建配置并启动转换（效果/音高参数统一从 GUI 状态读取，与实时一致）
        try:
            state = collect_params(self.window)
            config = OfflineParams(
                input_path=inp,
                output_path=out,
                voice=state.voice,
                f0=state.f0,
                buffer=state.buffer,
                audio=state.audio,
                texture=state.texture,
                rms_mix=state.rms_mix,
                volume=state.volume,
                model_path=pth,
                hubert=self.window.hubert_combo.currentText(),
            )
        except Exception as exc:
            logger.error("离线转换初始化失败:\n%s", traceback.format_exc())
            self.window._show_error(f"离线转换初始化失败: {exc}")
            return

        self.worker = OfflineWorker(config, parent=self.window)
        self._converting = True
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

        self.window.offline_button.setEnabled(False)
        self.window.offline_button.setText("转换中...")
        self.window.offline_cancel_btn.setEnabled(True)
        self.window.offline_status.setText("初始化...")
        self.window.offline_progress.setValue(0)

    def cancel_conversion(self) -> None:
        """请求取消当前离线转换（worker 在下一个块检查时中止）。"""
        if self.worker and self._converting:
            self.worker.request_cancel()
            self.window.offline_cancel_btn.setEnabled(False)
            self.window.offline_status.setText("正在取消...")

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """退出程序时的清理：请求取消并等待 worker 结束。

        OfflineWorker 是 QThread（非 daemon），run() 里可能正卡在 CUDA 推理或
        模型加载中；不等待就退出会让进程挂起（「点了退出没反应」）。
        取消标志在下一个块边界被检查，故正常情况零点几秒即返回；
        仍在加载模型等长阻塞阶段时可能等到 timeout_ms，超时后照常退出，
        不让退出流程无限期卡住。
        """
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.request_cancel()
            if not worker.wait(timeout_ms):
                logger.warning("离线转换线程 %dms 内未结束（可能卡在模型加载/CUDA 调用中），继续退出", timeout_ms)
        self.worker = None
        self._converting = False

    def _on_progress(self, current: int, total: int) -> None:
        """更新进度。"""
        self.window.offline_status.setText(f"进度: {current}/{total}")
        if total > 0:
            self.window.offline_progress.setValue(current)

    def _reset_conversion_ui(self, status_text: str) -> None:
        """转换结束（完成/取消/出错）共用的 UI 复位。"""
        self._converting = False
        self.window.offline_button.setEnabled(True)
        self.window.offline_button.setText("开始转换")
        self.window.offline_cancel_btn.setEnabled(False)
        self.window.offline_status.setText(status_text)

    def _on_finished(self, path: str) -> None:
        """转换完成 — worker 会自动退出并由 Qt 事件循环回收，不阻塞等待。"""
        self._reset_conversion_ui("完成")
        # QThread.run() 返回后线程自动结束，worker 对象稍后由 Qt 事件循环删除。
        # 不调用 deleteLater()/wait()，避免阻塞和 "Destroyed while thread still running" 警告。

    def _on_cancelled(self) -> None:
        """转换被用户取消。"""
        self._reset_conversion_ui("已取消")

    def _on_error(self, msg: str) -> None:
        """转换出错。"""
        self._reset_conversion_ui("错误")
        # 与 _on_finished 一致：error 信号在 run() 内发出，此刻线程可能尚未
        # 真正退出，提前析构有竞态。引用保留到下次 start_conversion 被新 worker 覆盖。
        self.window._show_error(f"离线推理错误: {msg}")


class OfflineWorker(QThread):
    """离线推理线程：模拟播放→转换→写录，复用实时引擎全链路。

    信号名说明：完成信号叫 result_ready 而非 finished——finished 会遮蔽
    QThread 的内建 C++ 信号，两种语义分叉后极易接错（内建 finished() 表示
    线程结束、无参数；本信号携带输出路径且发出时线程可能尚未退出）。
    """

    progress = Signal(int, int)
    result_ready = Signal(str)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, params: OfflineParams, parent=None):
        super().__init__(parent)
        self.params = params
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """请求取消（线程安全：bool 赋值在 GIL 下原子）。"""
        self._cancel_requested = True

    def run(self):
        import torch  # 惰性导入，避免 GUI 启动时加载 torch

        try:
            self._do_run()
        except CancelledError:
            self.cancelled.emit()
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
        from rvc.streaming.engine import VoiceEngine

        self.progress.emit(0, 100)
        engine = VoiceEngine(self.params)
        engine.load_model(self.params.model_path, hubert=self.params.hubert)
        self.progress.emit(20, 100)

        def _progress(cur, total):
            # 推理进度占 80%（模型加载占前 20%），避免进度回退
            pct = 20 + int(cur * 80 / total) if total > 0 else 20
            self.progress.emit(pct, 100)

        def _cancel_check():
            return self._cancel_requested

        engine.process_file(self.params, progress_cb=_progress, cancel_cb=_cancel_check)
        self.progress.emit(100, 100)
        self.result_ready.emit(self.params.output_path)
