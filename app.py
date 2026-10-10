"""RVC — 实时语音转换工具

用法:
    python app.py --infer    启动推理 GUI
    python app.py --train    启动训练命令行向导（Windows / Linux 通用）
"""
import argparse
import io
import logging
import os
import sys

if sys.stdout is not None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

LOG_FORMAT = "%(message)s"


def _configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(stream=sys.stdout)] if sys.stdout else [],
        force=True,
    )
    for name in ("torchfcpe", "numba", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)


def main():
    parser = argparse.ArgumentParser(description="RVC 实时语音转换工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--infer", action="store_true", help="启动推理 GUI")
    group.add_argument("--train", action="store_true", help="启动训练命令行向导")
    # 训练向导自己解释额外参数（第一个非选项参数 = 数据集路径），
    # 故这里用 parse_known_args 把其余参数原样透传，而不是由入口解析。
    args, extra = parser.parse_known_args()

    if args.train:
        from rvc.train.cli.run import main as train_main

        sys.exit(train_main(extra))

    from PySide6.QtWidgets import QApplication

    from gui.styles import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)

    from gui.infer.view.window import MainWindow

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
