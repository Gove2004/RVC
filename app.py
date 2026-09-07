"""RVC — 实时语音转换工具

用法:
    python app.py --infer    启动推理 GUI
    python app.py --train    启动训练 GUI
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
    group.add_argument("--train", action="store_true", help="启动训练 GUI")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication
    from gui.styles import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)

    if args.infer:
        from gui.infer.view.main_window import MainWindow
        win = MainWindow()
    else:
        from gui.train.window import TrainWindow
        win = TrainWindow()

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
