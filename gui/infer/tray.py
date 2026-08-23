"""系统托盘 — 最小化到任务栏右下角图标。

托盘不可用时直接抛异常（不 fallback），由调用方提示并退出。
图标首次运行程序生成并缓存到 assets/icon.png，之后直接加载，避免重复绘制。
"""
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

logger = logging.getLogger(__name__)

ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "icon.png"


def _draw_icon() -> QIcon:
    """程序内绘制托盘图标：深色圆底 + 三根声波条。"""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(40, 40, 40))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    pen = QPen(QColor(66, 133, 244), 6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    bars = [(18, 8), (29, 22), (40, 14)]
    for x, h in bars:
        p.drawLine(x, 32 - h // 2, x, 32 + h // 2)
    p.end()
    return QIcon(pm)


def _load_or_make_icon() -> QIcon:
    """优先加载缓存的图标文件；不存在则绘制并保存（仅首次有绘制开销）。"""
    if ICON_PATH.exists():
        return QIcon(str(ICON_PATH))
    icon = _draw_icon()
    try:
        icon.pixmap(64, 64).save(str(ICON_PATH))
        logger.info("托盘图标已生成并缓存: %s", ICON_PATH)
    except Exception as e:
        logger.warning("托盘图标缓存失败（本次用内存绘制）: %s", e)
    return icon


class TrayManager:
    """托盘图标 + 右键菜单（状态区/显示/开始停止/直通/退出）。"""

    def __init__(self, window, on_quit=None):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("系统托盘不可用，无法最小化到托盘")
        self.window = window
        self.on_quit = on_quit
        self._notified = False

        self.tray = QSystemTrayIcon(_load_or_make_icon(), window)
        self.tray.setToolTip("RVC 实时变声")

        menu = QMenu()
        act_show = QAction("显示主窗口", menu)
        act_show.triggered.connect(self.show_window)
        self.act_toggle = QAction("开始变声", menu)
        self.act_toggle.triggered.connect(self._toggle_running)
        act_quit = QAction("退出", menu)
        act_quit.triggered.connect(self.quit)

        menu.addAction(act_show)
        menu.addAction(self.act_toggle)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()
        logger.info("系统托盘已启用")

    def _on_activated(self, reason):
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_window()

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _toggle_running(self):
        """托盘开始/停止变声（不打开窗口）。"""
        eng = self.window.controller._engine
        if eng is not None and eng.running:
            self.window._stop()
        else:
            self.window._start()

    def update_status(self):
        """从引擎读取状态，刷新 tooltip 与开始/停止文本。"""
        eng = self.window.controller._engine
        running = eng is not None and eng.running
        latency = f"{eng.measure_ms:.0f}ms" if running and eng.measure_ms > 0 else "-"
        state = "推理中" if running else "已停止"
        self.tray.setToolTip(f"RVC 实时变声 状态: {state} 延迟: {latency}")
        self.act_toggle.setText("停止变声" if running else "开始变声")

    def notify_minimized(self):
        """首次隐藏到托盘时气泡提示一次。"""
        if self._notified:
            return
        self._notified = True
        QTimer.singleShot(
            300,
            lambda: self.tray.showMessage(
                "RVC 实时变声",
                "已最小化到托盘，点击图标恢复窗口",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            ),
        )

    def quit(self):
        self.tray.hide()
        if self.on_quit:
            self.on_quit()
