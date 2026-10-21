"""离线推理管理器 — 负责离线音频文件转换（含输出文件监听/删除）。"""

import ctypes
import logging
import os
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QThread, Signal

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


def _recycle_file(path: str) -> None:
    """把文件移入回收站（Windows SHFileOperation，FOF_ALLOWUNDO）。

    不用 os.remove——那是永久删除、不可恢复；回收站删除给误删留退路。
    非 Windows 平台抛 NotImplementedError（本应用仅 Windows）。
    """
    if os.name != "nt":
        raise NotImplementedError("回收站删除仅支持 Windows")

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", ctypes.c_int),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400

    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    # c_wchar_p 会在末尾补 \0，字符串自带一个 \0 ⇒ 恰为双零结尾（列表语义）
    op.pFrom = path + "\0"
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    ret = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if ret != 0 or op.fAnyOperationsAborted:
        raise OSError(f"SHFileOperationW 返回 {ret}")


class OfflineConversion:
    """管理离线音频文件转换流程。"""

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.worker = None
        self._converting = False
        self._preview_timer = None

    @property
    def is_converting(self) -> bool:
        """是否有离线转换在跑（供实时侧做互斥守卫）。"""
        return self._converting

    def toggle_conversion(self) -> None:
        """开始/取消单按钮的调度：空闲时开始转换，转换中请求取消。"""
        if self._converting:
            self.cancel_conversion()
        else:
            self.start_conversion()

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
        # from_inference 整体拷贝——逐字段手抄会漏新字段（protect 漏传事故）
        try:
            state = collect_params(self.window)
            config = OfflineParams.from_inference(
                state,
                input_path=inp,
                output_path=out,
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

        # 转换会覆写输出文件：先停监听，监听/删除按钮置灰直到转换结束
        self.stop_preview()
        self.window.offline_preview_btn.setEnabled(False)
        self.window.offline_delete_btn.setEnabled(False)
        self.window.offline_button.setText("取消")
        self.window.offline_status.setText("初始化...")
        self.window.offline_progress.setValue(0)

    def cancel_conversion(self) -> None:
        """请求取消当前离线转换（worker 在下一个块检查时中止）。"""
        if self.worker and self._converting:
            self.worker.request_cancel()
            self.window.offline_button.setEnabled(False)
            self.window.offline_button.setText("正在取消...")
            self.window.offline_status.setText("正在取消...")

    # ── 输出文件监听 / 删除 ──

    def _output_path(self) -> str:
        """由当前输入文件派生输出路径（未选输入返回空串）。"""
        inp = self.window.offline_input_btn.path().strip()
        return default_output_path(inp)

    def _preview_active(self) -> bool:
        import sounddevice as sd
        try:
            s = sd.get_stream()
        except RuntimeError:
            return False  # 尚未调用过 play()，必然未在播放
        return s is not None and s.active

    def toggle_preview(self) -> None:
        """监听输出文件：播放 / 再点停止（按钮文本随状态切换）。"""
        import sounddevice as sd
        import soundfile as sf

        if self._preview_active():
            sd.stop()
            self._stop_preview_ui()
            return

        path = self._output_path()
        if not path:
            self.window._show_warning("请先选择输入文件")
            return
        if not os.path.exists(path):
            self.window._show_warning("输出文件不存在，请先完成一次转换")
            return
        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as exc:
            self.window._show_error(f"无法读取音频文件: {exc}")
            return
        if data.ndim == 1:
            data = data[:, None]
        try:
            sd.play(data, sr)
        except Exception as exc:
            self.window._show_error(f"播放失败: {exc}")
            return

        self.window.offline_preview_btn.setText("停止")
        self.window.offline_status.setText(f"监听中: {os.path.basename(path)}")
        if self._preview_timer is None:
            self._preview_timer = QTimer(self.window)
            self._preview_timer.setInterval(150)
            self._preview_timer.timeout.connect(self._poll_preview)
        self._preview_timer.start()

    def _poll_preview(self) -> None:
        """播放自然结束后复位按钮（sounddevice 的 play 流是自动关闭的）。"""
        if not self._preview_active():
            self._stop_preview_ui()

    def _stop_preview_ui(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self.window.offline_preview_btn.setText("监听")

    def stop_preview(self) -> None:
        """外部强制停止监听（开始转换 / 删除前调用，任意状态安全）。"""
        if self._preview_active():
            import sounddevice as sd
            sd.stop()
        self._stop_preview_ui()

    def delete_output(self) -> None:
        """删除输出文件——二次确认后移入回收站（可恢复）。"""
        from PySide6.QtWidgets import QMessageBox

        path = self._output_path()
        if not path:
            self.window._show_warning("请先选择输入文件")
            return
        if not os.path.exists(path):
            self.window._show_warning("输出文件不存在，无需删除")
            return

        ans = QMessageBox.question(
            self.window, "删除输出文件",
            f"确定删除该输出文件吗？\n{path}\n\n文件将移入回收站。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        self.stop_preview()
        try:
            _recycle_file(path)
        except Exception as exc:
            self.window._show_error(f"删除失败: {exc}")
            return
        self.window.offline_status.setText("输出文件已移入回收站")

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
        self.window.offline_button.setText("开始")
        self.window.offline_preview_btn.setEnabled(True)
        self.window.offline_delete_btn.setEnabled(True)
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
