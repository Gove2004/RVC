"""系统托盘 — 最小化到任务栏右下角图标。

托盘不可用时直接抛异常（不 fallback），由调用方提示并退出。
运行状态用图标底色表达：绿色=推理中，红色=已停止。
图标首次运行程序生成并缓存到 assets/，之后直接加载，避免重复绘制。
"""
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
ICON_IDLE_PATH = ASSETS_DIR / "icon_idle.png"      # 红：未运行
ICON_ACTIVE_PATH = ASSETS_DIR / "icon_active.png"  # 绿：推理中


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
    bars = [(21, 8), (32, 22), (43, 14)]  # 三条杠以圆中心 x=32 对称（原 18/29/40 偏左 3px）
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
        logger.info("托盘图标已生成并缓存: %s", path)
    except Exception as e:
        logger.warning("托盘图标缓存失败（本次用内存绘制）: %s", e)
    return icon


class TrayManager:
    """托盘图标（双色状态）+ 右键菜单（开始/停止、显示、退出）。"""

    def __init__(self, window, on_quit=None):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("系统托盘不可用，无法最小化到托盘")
        self.window = window
        self.on_quit = on_quit
        self._notified = False

        self._icon_idle = _load_or_make_icon(ICON_IDLE_PATH, QColor(211, 47, 47))      # 红
        self._icon_active = _load_or_make_icon(ICON_ACTIVE_PATH, QColor(46, 125, 50))  # 绿
        self.tray = QSystemTrayIcon(self._icon_idle, window)
        self.tray.setToolTip("-")

        menu = QMenu()
        act_show = QAction("显示", menu)
        act_show.triggered.connect(self.show_window)
        self.act_toggle = QAction("开始", menu)
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
        """从引擎读取状态，刷新：图标底色（绿=运行/红=停止）、tooltip 延迟、开始/停止文本。"""
        eng = self.window.controller._engine
        running = eng is not None and eng.running
        latency = f"{eng.measure_ms:.0f}ms" if running and eng.measure_ms > 0 else "-"
        self.tray.setToolTip(latency)
        self.tray.setIcon(self._icon_active if running else self._icon_idle)
        self.act_toggle.setText("停止" if running else "开始")

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
