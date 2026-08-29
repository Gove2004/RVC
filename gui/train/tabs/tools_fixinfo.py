"""训练工具 Tab — 修正模型信息（真名）（从 tools_tab.py 拆出）"""
from pathlib import Path

from PySide6.QtWidgets import (
    QGroupBox, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox,
)

from gui.train.tabs.tools_inspect import _run_inspect
from gui.train.widgets import ToolThread, browse_file
from gui.styles import ButtonStyles, Layout


def build_group(win) -> QGroupBox:
    group = QGroupBox("修正模型信息（真名）")
    grid = QGridLayout(group)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)

    win.fix_path = QLineEdit()
    btn_browse = QPushButton("浏览")
    btn_browse.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_browse.setStyleSheet(ButtonStyles.small())
    btn_browse.clicked.connect(lambda: browse_file(win, win.fix_path))
    grid.addWidget(QLabel("模型文件"), 0, 0)
    grid.addWidget(win.fix_path, 0, 1)
    grid.addWidget(btn_browse, 0, 2)

    win.fix_info = QLineEdit()
    win.fix_info.setPlaceholderText("输入新的真名，例如 exp01（自动去掉误带的 .pth）")
    btn_apply = QPushButton("应用")
    btn_apply.setFixedWidth(Layout.BTN_WIDTH_SMALL)
    btn_apply.setStyleSheet(ButtonStyles.small())
    btn_apply.clicked.connect(lambda: _run_fixinfo(win))
    grid.addWidget(QLabel("新真名"), 1, 0)
    grid.addWidget(win.fix_info, 1, 1)
    grid.addWidget(btn_apply, 1, 2)

    return group


def _run_fixinfo(win):
    from rvc.train.ckpt_utils import change_info

    path = win.fix_path.text().strip()
    info = win.fix_info.text().strip()
    if not path:
        QMessageBox.warning(win, "提示", "请选择模型文件")
        return
    if not info:
        QMessageBox.warning(win, "提示", "请输入新的真名")
        return
    if not Path(path).exists():
        QMessageBox.warning(win, "提示", "文件不存在")
        return
    if win._tool_thread and win._tool_thread.isRunning():
        win._tool_thread.wait()
    win._tool_thread = ToolThread(change_info, path, info)
    win._tool_thread.done.connect(lambda ok, msg: _on_fixinfo_done(win, ok, msg, path))
    win._tool_thread.start()


def _on_fixinfo_done(win, success, message, path):
    if success:
        QMessageBox.information(win, "完成", "模型信息已更新")
        # 若在检视框里也选了同一文件，刷新显示
        if Path(win.inspect_path.text().strip()).resolve() == Path(path).resolve():
            _run_inspect(win)
    else:
        QMessageBox.critical(win, "失败", message)
