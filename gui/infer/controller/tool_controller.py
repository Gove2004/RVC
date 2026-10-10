"""推理 GUI 的模型工具控制器 — 模型合并 / 检视 / 改 zip 原名 / 修正 info。

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

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.done.emit(True, result if isinstance(result, str) else "操作完成")
        except Exception as e:
            self.done.emit(False, str(e))


class ModelToolController:
    """模型文件工具：合并 / 检视 / 改 zip 原名 / 修正 info。"""

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

    def change_archive_name(self, path: str, new_name: str) -> str:
        from rvc.train.checkpoint import change_archive_name

        return change_archive_name(path, new_name)

    def change_model_info(self, path: str, model_info: str) -> None:
        from rvc.train.checkpoint import change_model_info

        change_model_info(path, model_info)
