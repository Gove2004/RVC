"""离线 Tab — 离线音频文件转换"""
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QGridLayout, QProgressBar,
)
from gui.controller.offline_controller import default_output_path
from gui.styles import ButtonStyles, Layout
from gui.styles.colors import Colors
from gui.view.widgets import ElidedLabel, PathPickerButton

AUDIO_FILTER = "音频 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus)"

# 未选输入文件时输出行的占位文案
OUTPUT_PLACEHOLDER = "选择输入文件后自动确定"


def _on_input_picked(win, path: str) -> None:
    """选完输入文件后刷新只读的输出路径（恒为输入同目录的 <原名>_converted.wav）。"""
    win.offline_output.setText(default_output_path(path) or OUTPUT_PLACEHOLDER)


def _readonly_style() -> str:
    """只读文本行的外观：与路径按钮一致，但文字为次要色以示不可编辑。"""
    return (
        f"background-color: {Colors.INPUT_BG};"
        f"color: {Colors.TEXT_SECONDARY};"
        f"border: 1px solid {Colors.BORDER};"
        f"border-radius: {Layout.RADIUS_NORMAL}px;"
        "padding: 3px 8px;"
    )


def build_offline_tab(win):
    """6 行布局：输入 / 输出 / 开始(取消) / 监听删除 / 进度 / 状态。"""
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    r = 0

    # 输入文件行（自带文件对话框的路径按钮，文本显示所选文件名）
    win.offline_input_btn = PathPickerButton(
        win, title="选择输入文件", filter_str=AUDIO_FILTER,
        placeholder="选择输入文件...",
    )
    win.offline_input_btn.pathChanged.connect(lambda p: _on_input_picked(win, p))
    g.addWidget(QLabel("输入文件"), r, 0)
    g.addWidget(win.offline_input_btn, r, 1, 1, 3)
    r += 1

    # 输出文件行（纯文本展示，不可修改：恒为输入文件同目录的 <原名>_converted.wav）
    win.offline_output = ElidedLabel(OUTPUT_PLACEHOLDER)
    win.offline_output.setMinimumHeight(28)
    win.offline_output.setStyleSheet(_readonly_style())
    g.addWidget(QLabel("输出文件"), r, 0)
    g.addWidget(win.offline_output, r, 1, 1, 3)
    r += 1

    # 开始 / 取消（单按钮切换：空闲时开始，转换中取消）
    win.offline_button = QPushButton("开始")
    win.offline_button.setMinimumHeight(32)
    win.offline_button.setToolTip("开始离线转换（输入 → <原名>_converted.wav）；转换中点击取消")
    win.offline_button.setStyleSheet(ButtonStyles.primary())
    win.offline_button.clicked.connect(win._off_toggle)
    g.addWidget(win.offline_button, r, 0, 1, 4)
    r += 1

    # 监听 / 删除（针对输出文件，无需打开文件管理器即可核对效果）
    win.offline_preview_btn = QPushButton("监听")
    win.offline_preview_btn.setMinimumHeight(32)
    win.offline_preview_btn.setToolTip("播放输出文件（播放中再点一次停止）")
    win.offline_preview_btn.clicked.connect(win.offline_conversion.toggle_preview)
    g.addWidget(win.offline_preview_btn, r, 0, 1, 2)

    win.offline_delete_btn = QPushButton("删除")
    win.offline_delete_btn.setMinimumHeight(32)
    win.offline_delete_btn.setToolTip("删除输出文件（移入回收站）")
    win.offline_delete_btn.setStyleSheet(ButtonStyles.danger())
    win.offline_delete_btn.clicked.connect(win.offline_conversion.delete_output)
    g.addWidget(win.offline_delete_btn, r, 2, 1, 2)
    r += 1

    # 进度条
    win.offline_progress = QProgressBar()
    win.offline_progress.setMinimumHeight(24)
    g.addWidget(win.offline_progress, r, 0, 1, 4)
    r += 1

    # 状态文本（最后一行）
    win.offline_status = QLabel("")
    g.addWidget(win.offline_status, r, 0, 1, 4)

    return w
