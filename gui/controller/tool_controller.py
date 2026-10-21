"""推理 GUI 的模型工具控制器 — 模型合并 / 检视。

原挂在训练 GUI 的 `TrainController` 上；训练改为命令行向导后，是 GUI 侧保留的
唯一「非推理」功能，故迁到推理端。耗时操作由 `ToolRunner` 在后台线程执行，
本类方法即其执行体；View 层不直接 import rvc，业务函数一律经此类转发。
"""
import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class ToolRunner(QThread):
    """后台执行模型合并 / 检视等耗时操作。"""

    done = Signal(bool, str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.done.emit(True, result if isinstance(result, str) else "操作完成")
        except Exception as e:
            self.done.emit(False, str(e))


def run_tool_async(win, fn, *args, on_done, wait_timeout_ms: int = 10000) -> None:
    """统一工具执行入口（融合/详情页签共用）：串行化后台执行。

    - 前一个任务未结束时等待（上限 wait_timeout_ms，避免无超时 wait 阻塞主线程）；
      超时则取消本次任务并把原因回调给调用方，而不是启动并发写同一模型文件
    - ToolRunner 以 win 为 parent，替换前对旧对象 deleteLater（不依赖 GC 回收）
    """
    old = win.tool_runner
    if old is not None:
        if old.isRunning() and not old.wait(wait_timeout_ms):
            logger.warning("模型工具线程 %dms 内未结束，本次操作取消", wait_timeout_ms)
            on_done(False, "上一次操作尚未结束，请稍后重试")
            return
        old.deleteLater()
    win.tool_runner = ToolRunner(fn, *args, parent=win)
    win.tool_runner.done.connect(on_done)
    win.tool_runner.start()


class ModelToolController:
    """模型文件工具：合并 / 检视。"""

    def merge_output_path(self, name: str) -> str:
        """合并输出的完整路径（assets/models/<name>.pth）。"""
        from rvc.runtime.paths import MODELS_DIR

        return str(MODELS_DIR / f"{name}.pth")

    def merge_models(self, path_a: str, path_b: str, ratio: float, out_path: str) -> None:
        from rvc.train.checkpoint import merge_models

        merge_models(path_a, path_b, ratio, out_path)

    def inspect_model(self, path: str) -> str:
        from rvc.train.checkpoint import inspect_model

        return inspect_model(path)
