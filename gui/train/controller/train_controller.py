"""训练控制器 — 训练 GUI 的业务逻辑收口（D5：三个工具 Tab 走 controller）。

View 层（gui/train/view/*）禁止直接 import rvc.*；
采样率→底模路径映射、模型合并/检视/改名等业务操作统一经本类。
耗时操作仍由 view 层 ToolThread 在后台线程执行，本类方法是其执行体。
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainController:
    """训练窗口控制器：底模匹配、模型工具、参数校验。"""

    # ── 底模匹配 ──

    def pretrained_paths(self, sr_text: str) -> tuple[str, str]:
        """采样率（"40k"/"48k"）→ 预训练 G/D 路径。

        存在即返回路径字符串，不存在返回空串（= 从零训练）。
        """
        from rvc.runtime.paths import PRETRAINED_ROOT

        paths = []
        for key in ("G", "D"):
            path = PRETRAINED_ROOT / f"f0{key}{sr_text}.pth"
            paths.append(str(path) if path.exists() else "")
        return paths[0], paths[1]

    # ── 模型工具 ──

    def merge_models(self, path_a: str, path_b: str, ratio: float, out_path: str) -> str:
        """合并两个模型，返回成功消息。耗时操作（ToolThread 后台执行）。"""
        from rvc.train.checkpoint import merge_models

        return merge_models(path_a, path_b, ratio, out_path)

    def merge_output_path(self, name: str) -> str:
        """合并输出的完整路径（assets/models/<name>.pth）。"""
        from rvc.runtime.paths import MODELS_DIR

        return str(MODELS_DIR / f"{name}.pth")

    def inspect_model(self, path: str) -> str:
        """查看模型信息（zip 原名/训练参数）。耗时操作（ToolThread 后台执行）。"""
        from rvc.train.checkpoint import inspect_model

        return inspect_model(path)

    def change_archive_name(self, path: str, new_name: str) -> str:
        """修改模型 zip 原名。耗时操作（ToolThread 后台执行）。"""
        from rvc.train.checkpoint import change_archive_name

        return change_archive_name(path, new_name)

    def change_model_info(self, path: str, model_info: str) -> str:
        """修正模型信息（真名）。耗时操作（ToolThread 后台执行）。"""
        from rvc.train.checkpoint import change_model_info

        return change_model_info(path, model_info)

    # ── 参数校验 ──

    def validate_options(self, raw: dict) -> "TrainRequest":
        """校验并整理训练参数（View 收集原始控件值，业务校验在此）。

        Raises:
            ValueError: 参数不合法（文案与旧版一致）。
        """
        exp_name = raw["exp_name"]
        input_dir = raw["input_dir"]
        if not exp_name:
            raise ValueError("实验名不能为空")
        if not input_dir or not Path(input_dir).exists():
            raise ValueError("请选择有效的音频目录")
        try:
            lr = float(raw["learning_rate"])
        except ValueError as exc:
            raise ValueError("学习率格式不正确") from exc
        from gui.train.controller.workers import TrainRequest
        return TrainRequest(
            exp_name=exp_name,
            input_dir=input_dir,
            sr=raw["sr"],
            epochs=raw["epochs"],
            batch_size=raw["batch_size"],
            save_every_epoch=raw["save_every_epoch"],
            learning_rate=lr,
            pretrain_g=raw["pretrain_g"],
            pretrain_d=raw["pretrain_d"],
            hubert=raw["hubert"],
        )
