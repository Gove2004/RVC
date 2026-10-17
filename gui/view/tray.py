"""系统托盘 — 最小化到任务栏右下角图标。

托盘不可用时直接抛异常（不 fallback），由调用方提示并退出。
输出状态用图标底色表达：绿色=变声输出中，蓝色=原声直通中，红色=停止输出。
图标首次运行程序生成并缓存到 assets/，之后直接加载，避免重复绘制。
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

if TYPE_CHECKING:
    from gui.view.window import MainWindow

logger = logging.getLogger(__name__)


def _draw_icon(base_color: QColor) -> QIcon:
    """程序内绘制托盘图标：圆底（底色区分状态）+ 白色声波条。"""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(base_color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    pen = QPen(QColor(255, 255, 255), 6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    bars = [(21, 8), (32, 22), (43, 14)]  # 三条杠以圆中心 x=32 对称
    for x, h in bars:
        p.drawLine(x, 32 - h // 2, x, 32 + h // 2)
    p.end()
    return QIcon(pm)


def _load_or_make_icon(path: Path, base_color: QColor) -> QIcon:
    """优先加载缓存图标文件；不存在则绘制并保存（仅首次有绘制开销）。"""
    if path.exists():
        return QIcon(str(path))
    icon = _draw_icon(base_color)
    try:
        icon.pixmap(64, 64).save(str(path))
        logger.info("托盘图标已生成并缓存：%s", path)
    except Exception as e:
        logger.warning("托盘图标缓存失败（本次用内存绘制）：%s", e)
    return icon


class SystemTray:
    """托盘图标（三色状态：绿=变声/蓝=直通/红=停止）+ 右键菜单（显示、退出）。

    三态图标文件路径由调用方经 controller 下发（View 层不直接 import rvc 路径常量）。
    """

    def __init__(self, window: "MainWindow", on_quit=None, icon_idle: Path | None = None,
                 icon_active: Path | None = None, icon_passthrough: Path | None = None):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("系统托盘不可用，无法最小化到托盘")
        self.window = window
        self.on_quit = on_quit
        self._notified = False
        # 单击循环切档：停止→变声→停止→原声→停止……
        # 唯一状态 = 下次从停止档启动的档位；由「离开的档位」推导
        # （离开变声/加载 → 下次开原声；离开原声 → 下次开变声），
        # 启动失败不消耗交替（下次单击重试同档）。
        self._next_from_stop = "convert"

        self._icon_idle = _load_or_make_icon(icon_idle, QColor(211, 47, 47))        # 红
        self._icon_active = _load_or_make_icon(icon_active, QColor(46, 125, 50))    # 绿
        self._icon_pass = _load_or_make_icon(icon_passthrough, QColor(21, 101, 192))  # 蓝
        self.tray = QSystemTrayIcon(self._icon_idle, window)
        self.tray.setToolTip("-")

        menu = QMenu()
        act_show = QAction("显示", menu)
        act_show.triggered.connect(self.show_window)
        act_quit = QAction("退出", menu)
        act_quit.triggered.connect(self.quit)

        menu.addAction(act_show)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()
        logger.info("系统托盘已启用")

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # 单击：循环切档（停止→变声→停止→原声→停止），双击不响应避免弹窗
            self._toggle_running()

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _toggle_running(self):
        """托盘单击循环切档（不打开窗口）：
        停止→变声→停止→原声→停止（从停止档启动的档位交替轮换）。
        加载中/变声中/原声中单击一律回停止档。"""
        win = self.window
        if win._loading or win.controller.is_running or win.controller.passthrough_running:
            left_convert = win._loading or win.controller.is_running
            self._next_from_stop = "passthrough" if left_convert else "convert"
            win._on_mode_stop()
        elif self._next_from_stop == "passthrough":
            win._on_mode_passthrough()
        else:
            win._on_mode_convert()

    def update_status(self):
        """刷新三态图标与 tooltip：绿=变声中 / 蓝=原声直通中 / 红=停止输出。"""
        snap = self.window.controller.snapshot()
        if snap.running:
            icon, tip = self._icon_active, (f"{snap.measure_ms:.0f}ms" if snap.measure_ms > 0 else "-")
        elif self.window.controller.passthrough_running:
            icon, tip = self._icon_pass, "原声直通中"
        else:
            icon, tip = self._icon_idle, "-"
        self.tray.setToolTip(tip)
        self.tray.setIcon(icon)

    def notify_minimized(self):
        """首次隐藏到托盘时气泡提示一次。"""
        if self._notified:
            return
        self._notified = True
        QTimer.singleShot(
            300,
            lambda: self.tray.showMessage(
                "RVC",
                "已最小化到托盘，点击图标恢复窗口",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            ),
        )

    def quit(self):
        self.tray.hide()
        if self.on_quit:
            self.on_quit()
