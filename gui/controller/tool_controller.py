"""推理 GUI 的模型工具控制器 — 模型合并 / 检视。

原挂在训练 GUI 的 `TrainController` 上；训练改为命令行向导后，是 GUI 侧保留的
唯一「非推理」功能，故迁到推理端。耗时操作由 `ToolRunner` 在后台线程执行，
本类方法即其执行体；View 层不直接 import rvc，业务函数一律经此类转发。

队列接口（`tool_runner` / `tool_queue`）同时被生命周期用于退出前等待，
两处字段属于 `gui.view.contracts.InferWindowHost`，不要在别处另起一份。
"""
from PySide6.QtCore import QThread, Signal


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


def run_tool_async(win, fn, *args, on_done) -> None:
    """统一工具执行入口（融合/详情页签共用）：后台排队串行执行。

    队列存在 win.tool_queue（契约见 gui/view/contracts.py）：
    - 当前没有任务在跑 → 立刻启动
    - 已有任务在跑 → 排队，等它结束再启动；**主线程一次也不等**

    旧实现用 `old.wait(10000)` 在 GUI 线程上等上一个工具线程结束，最坏
    僵住 10 秒（Windows 上还会被判定「无响应」）。串行化本身是必要的
    （合并与检视都要读同一批模型文件），但等待不该发生在主线程。
    """
    queue = getattr(win, "tool_queue", None)
    if queue is None:
        queue = win.tool_queue = []
    queue.append((fn, args, on_done))
    _drain_tool_queue(win)


def _drain_tool_queue(win) -> None:
    """队列泵：空闲则取出下一项启动；被 finished 回调驱动，永不阻塞。"""
    current = win.tool_runner
    if current is not None and current.isRunning():
        return
    queue = getattr(win, "tool_queue", None)
    if not queue:
        return

    fn, args, on_done = queue.pop(0)
    runner = ToolRunner(fn, *args, parent=win)
    runner.done.connect(on_done)
    # finished 在主线程排队投递：先释放本 runner，再泵下一项，避免
    # 「新任务覆盖 win.tool_runner 后旧对象仍存活」
    runner.finished.connect(lambda: _on_tool_finished(win, runner))
    win.tool_runner = runner
    runner.start()


def _on_tool_finished(win, runner) -> None:
    if win.tool_runner is runner:
        win.tool_runner = None
    runner.deleteLater()
    _drain_tool_queue(win)


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
