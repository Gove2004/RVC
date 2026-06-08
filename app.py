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

# Windows 中文终端修复
if sys.stdout is not None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)] if sys.stdout else [],
)

now_dir = os.getcwd()
sys.path.append(now_dir)


def _set_dark(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette, QColor

    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(40, 40, 40))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(30, 30, 30))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(55, 55, 55))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight, QColor(66, 133, 244))
    pal.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(pal)
    app.setStyleSheet(
        "QGroupBox{font-weight:bold;margin-top:6px}"
        "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px}"
        "QSlider{min-height:18px}"
        "QSlider::groove:horizontal{height:4px}"
        "QSlider::handle:horizontal{width:12px;margin:-5px 0}"
    )


def main():
    parser = argparse.ArgumentParser(description="RVC 实时语音转换工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--infer", action="store_true", help="启动推理 GUI")
    group.add_argument("--train", action="store_true", help="启动训练 GUI")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    _set_dark(app)

    if args.infer:
        from app.infer.window import MainWindow
        win = MainWindow()
    else:
        from app.train.window import TrainWindow
        win = TrainWindow()

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
