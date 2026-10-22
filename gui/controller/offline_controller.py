"""离线推理管理器 — 离线音频文件转换（含输出文件监听/删除）。

分层契约：本类是 controller，唯一 UI 依赖是 `gui.view.contracts.OfflineViewHost`
（MainWindow 实现）。禁止触碰任何控件——控件机械操作（置灰/文案/进度条/对话框）
全部经宿主语义方法完成，本类因此可无头测试（FakeHost 替身）。
"""

import logging
import os
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QThread, Signal

from gui.controller.recycle import recycle_file
from rvc.core.config import OfflineParams
from rvc.core.errors import CancelledError

if TYPE_CHECKING:
    from gui.view.contracts import OfflineViewHost

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
    """管理离线音频文件转换流程（controller — 只依赖 OfflineViewHost）。"""

    def __init__(self, host: "OfflineViewHost"):
        self.host = host
        self.worker = None
        self._converting = False
        self._preview_timer = None

    @property
    def is_converting(self) -> bool:
        """是否有离线转换在跑（供实时侧做互斥守卫）。"""
        return self._converting

    def start_conversion(self) -> None:
        """开始离线转换。"""
        if self._converting:
            self.host.show_warning("已有转换任务正在运行")
            return

        # 实时推理运行时禁止离线转换（共享模型缓存，同时运行会互相干扰 CUDA Graph）。
        # 经 controller.is_running 判断：引擎未构造时短路返回 False，不会触发惰性加载。
        if self.host.controller.is_running:
            self.host.show_warning("实时推理正在运行，请先停止实时推理再进行离线转换")
            return

        # 模型加载中也要拒：实时/离线共用同一个 engine，离线 worker 会在
        # engine._load_lock 上静默阻塞整个加载时长，进度条停在「初始化」，
        # 用户没有任何反馈（加载线程本身没有结束，is_running 仍为 False）。
        if self.host.is_engine_loading():
            self.host.show_warning("模型正在加载中，请等待加载完成再开始离线转换")
            return

        inp = self.host.offline_input_path()
        if not inp:
            self.host.show_warning("请选择输入文件")
            return
        if not os.path.exists(inp):
            self.host.show_warning(f"文件不存在: {inp}")
            return

        # 输出路径由输入派生（界面上是只读文本，用户不可修改）
        out = default_output_path(inp)

        pth = self.host.offline_model_path()
        if not pth:
            self.host.show_warning("请先在「音色」中选择模型文件")
            return

        # 构建配置并启动转换（效果/音高参数统一从 GUI 状态读取，与实时一致）
        # from_inference 整体拷贝——逐字段手抄会漏新字段（protect 漏传事故）
        try:
            config = OfflineParams.from_inference(
                self.host.collect_gui_state(),
                input_path=inp,
                output_path=out,
                hubert=self.host.offline_hubert_variant(),
            )
        except Exception as exc:
            logger.error("离线转换初始化失败:\n%s", traceback.format_exc())
            self.host.show_error(f"离线转换初始化失败: {exc}")
            return

        self.worker = OfflineWorker(
            config,
            engine_provider=lambda: self.host.controller.engine,
            parent=self.host,
        )
        self._converting = True
        # QThread 结束后不会自动销毁（无 WA_DeleteOnClose）；不挂 deleteLater
        # 就会在 MainWindow 名下累积 —— 每次转换泄漏一个含完整 OfflineParams
        # 的线程对象。接内建 finished（线程真正结束后由事件循环回收，不违反
        # _on_finished 注释里担心的「提前析构」竞态），不要接自定义 result_ready。
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

        # 转换会覆写输出文件：先停监听（按钮置灰等 UI 三态由 enter_converting 统一收）
        self.stop_preview()
        self.host.offline_enter_converting()

    def cancel_conversion(self) -> None:
        """请求取消当前离线转换（worker 在下一个块检查时中止）。"""
        if self.worker and self._converting:
            self.worker.request_cancel()
            self.host.offline_enter_cancelling()

    # ── 输出文件监听 / 删除 ──

    def _output_path(self) -> str:
        """由当前输入文件派生输出路径（未选输入返回空串）。"""
        return default_output_path(self.host.offline_input_path())

    def _preview_active(self) -> bool:
        import sounddevice as sd
        try:
            s = sd.get_stream()
        except RuntimeError:
            return False  # 尚未调用过 play()，必然未在播放
        return s is not None and s.active

    def toggle_preview(self) -> None:
        """监听输出文件：播放 / 再点停止（按钮文案由宿主切换）。"""
        import sounddevice as sd
        import soundfile as sf

        if self._preview_active():
            sd.stop()
            self._stop_preview_ui()
            return

        path = self._output_path()
        if not path:
            self.host.show_warning("请先选择输入文件")
            return
        if not os.path.exists(path):
            self.host.show_warning("输出文件不存在，请先完成一次转换")
            return
        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as exc:
            self.host.show_error(f"无法读取音频文件: {exc}")
            return
        if data.ndim == 1:
            data = data[:, None]
        try:
            sd.play(data, sr)
        except Exception as exc:
            self.host.show_error(f"播放失败: {exc}")
            return

        self.host.offline_set_previewing(True, name=os.path.basename(path))
        if self._preview_timer is None:
            self._preview_timer = QTimer(self.host)
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
        self.host.offline_set_previewing(False)

    def stop_preview(self) -> None:
        """外部强制停止监听（开始转换 / 删除前调用，任意状态安全）。"""
        if self._preview_active():
            import sounddevice as sd
            sd.stop()
        self._stop_preview_ui()

    def delete_output(self) -> None:
        """删除输出文件——二次确认（宿主弹窗）后移入回收站（可恢复）。"""
        path = self._output_path()
        if not path:
            self.host.show_warning("请先选择输入文件")
            return
        if not os.path.exists(path):
            self.host.show_warning("输出文件不存在，无需删除")
            return

        if not self.host.offline_confirm_delete(path):
            return

        self.stop_preview()
        try:
            recycle_file(path)
        except Exception as exc:
            self.host.show_error(f"删除失败: {exc}")
            return
        self.host.offline_set_status("输出文件已移入回收站")

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
        self.host.offline_set_progress(current, total)

    def _on_finished(self, path: str) -> None:
        """转换完成 — 线程由 finished→deleteLater 自行回收，这里只收敛状态。"""
        self._converting = False
        self.host.offline_reset("完成")
        # 不调用 wait()：result_ready 在 run() 内发出，此刻线程可能尚未退出，
        # 阻塞等待会冻住 UI；对象回收交给 start_conversion 里挂的
        # finished→deleteLater（线程真正结束后才触发）。

    def _on_cancelled(self) -> None:
        self._converting = False
        self.host.offline_reset("已取消")

    def _on_error(self, msg: str) -> None:
        """转换出错。"""
        self._converting = False
        self.host.offline_reset("错误")
        # 与 _on_finished 一致：error 信号在 run() 内发出，此刻线程可能尚未
        # 真正退出，故不在此处析构；回收由 start_conversion 挂的
        # finished→deleteLater 保证（线程结束后才触发）。
        self.host.show_error(f"离线推理错误: {msg}")


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

    def __init__(self, params: OfflineParams, engine_provider, parent=None):
        super().__init__(parent)
        self.params = params
        self._engine_provider = engine_provider
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
        # 复用实时共享引擎（engine_controller.engine，带 inference_cache）：
        # process_file 的设计契约就是离线/实时共用同一 engine（见 engine.process_file
        # 文档）。同 pth 时 load_model 命中快路径零加载，也避免与实时引擎并存导致
        # 的显存双份。互斥由 start_conversion 的 is_running 守卫保证。
        self.progress.emit(0, 100)
        engine = self._engine_provider()
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
