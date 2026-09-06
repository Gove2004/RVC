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


def _set_dark(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette, QColor

    app.setStyle("Fusion")

    # Windows 11 深色模式（夜晚）配色
    pal = QPalette()
    # 背景层
    pal.setColor(QPalette.Window, QColor(18, 18, 18))           # #121212 窗口背景
    pal.setColor(QPalette.Base, QColor(14, 14, 14))             # #0e0e0e 输入框/文本区
    pal.setColor(QPalette.AlternateBase, QColor(22, 22, 22))    # #161616 交替行
    pal.setColor(QPalette.ToolTipBase, QColor(28, 28, 28))      # #1c1c1c 工具提示
    pal.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    # 文字
    pal.setColor(QPalette.WindowText, QColor(255, 255, 255))
    pal.setColor(QPalette.Text, QColor(229, 229, 229))
    pal.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    pal.setColor(QPalette.BrightText, QColor(255, 255, 255))
    # 控件
    pal.setColor(QPalette.Button, QColor(28, 28, 28))            # #1c1c1c 按钮
    # 强调色（Windows Blue）
    pal.setColor(QPalette.Highlight, QColor(0, 120, 212))        # #0078D4
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.Link, QColor(76, 194, 255))            # #4cc2ff
    # 禁用态
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(109, 109, 109))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(109, 109, 109))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(109, 109, 109))
    pal.setColor(QPalette.Disabled, QPalette.Button, QColor(18, 18, 18))
    pal.setColor(QPalette.Disabled, QPalette.Base, QColor(14, 14, 14))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, QColor(42, 42, 42))

    app.setPalette(pal)
    app.setStyleSheet(
        # 原有样式
        "QGroupBox{font-weight:bold;margin-top:6px}"
        "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px}"
        "QSlider{min-height:18px}"
        "QSlider::groove:horizontal{height:4px}"
        "QSlider::handle:horizontal{width:12px;margin:-5px 0}"
        # Windows 11 风格控件
        "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QTextEdit"
        "{border:1px solid #2a2a2a;border-radius:4px;background-color:#0e0e0e;padding:3px 6px;selection-background-color:#0078D4;}"
        "QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus,QTextEdit:focus"
        "{border-color:#0078D4;}"
        "QComboBox::drop-down{border:none;width:22px;}"
        "QComboBox QAbstractItemView{border:1px solid #2a2a2a;background-color:#1c1c1c;selection-background-color:#0078D4;outline:none;}"
        "QTabWidget::pane{border:1px solid #2a2a2a;top:-1px;}"
        "QTabBar::tab{background-color:#121212;padding:6px 14px;border:1px solid #2a2a2a;border-bottom:none;}"
        "QTabBar::tab:selected{background-color:#1c1c1c;}"
        "QTabBar::tab:hover:!selected{background-color:#181818;}"
        "QScrollBar:vertical{background:#121212;width:10px;margin:0;}"
        "QScrollBar::handle:vertical{background:#2a2a2a;border-radius:5px;min-height:30px;}"
        "QScrollBar::handle:vertical:hover{background:#3a3a3a;}"
        "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        "QScrollBar:horizontal{background:#121212;height:10px;margin:0;}"
        "QScrollBar::handle:horizontal{background:#2a2a2a;border-radius:5px;min-width:30px;}"
        "QScrollBar::handle:horizontal:hover{background:#3a3a3a;}"
        "QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}"
        "QMenu{background-color:#1c1c1c;border:1px solid #2a2a2a;padding:4px;}"
        "QMenu::item{padding:5px 20px;border-radius:4px;}"
        "QMenu::item:selected{background-color:#0078D4;}"
        "QMenu::separator{height:1px;background:#2a2a2a;margin:4px 8px;}"
        "QCheckBox::indicator,QRadioButton::indicator{width:14px;height:14px;border:1px solid #555;border-radius:3px;background:#0e0e0e;}"
        "QCheckBox::indicator:checked,QRadioButton::indicator:checked{background:#0078D4;border-color:#0078D4;}"
        "QProgressBar{border:1px solid #2a2a2a;border-radius:4px;background:#0e0e0e;text-align:center;}"
        "QProgressBar::chunk{background-color:#0078D4;border-radius:3px;}"
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
        from gui.infer.window import MainWindow
        win = MainWindow()
    else:
        from gui.train.window import TrainWindow
        win = TrainWindow()

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
