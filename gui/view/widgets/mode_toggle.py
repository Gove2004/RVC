"""运行态切换控件 — 三档分段式单选（原声/停止/变声）。"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QPushButton, QHBoxLayout

from gui.styles import Layout
from gui.styles.colors import Colors


class ModeToggle(QWidget):
    """三档运行态切换 — 分段式单选控件（原声/停止/变声，相互独立互斥）。

    视觉：一条分段滑轨，当前档以自身状态色实心填充（蓝=原声 / 红=停止 /
    绿=变声，与托盘三色一致），非当前档暗底灰字。点目标档切档并发
    modeChanged 信号；点当前档无副作用（仅恢复勾选态）。

    业务无关：本控件只管档位视觉与信号，档位含义的执行（停引擎、开直通、
    加载模型）由宿主连接 modeChanged 自行分发；宿主用 set_mode() 回写
    实际状态（唯一事实源 = 实际流状态，控件勾选态只是其镜像）。
    """

    modeChanged = Signal(str)

    SEGMENTS = (
        ("passthrough", "原声", "原声直通：麦克风原样转发到主输出（不经模型），\n目标软件固定录虚拟声卡时用它临时说原声"),
        ("stop", "停止", "停止：关闭所有输出"),
        ("convert", "变声", "变声：加载模型并实时转换输出"),
    )
    _VALID = {"passthrough", "stop", "convert", "loading"}
    _COLORS = {"passthrough": Colors.INFO, "stop": Colors.DANGER, "convert": Colors.SUCCESS}

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._mode = "stop"
        self._buttons = {}
        for i, (mode, text, tip) in enumerate(self.SEGMENTS):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedHeight(Layout.BTN_HEIGHT_NORMAL)
            b.setToolTip(tip)
            b.clicked.connect(lambda checked=False, m=mode: self._on_clicked(m))
            lay.addWidget(b)
            self._buttons[mode] = b
            # 分段圆角：首段圆左、末段圆右、中段直角（一体滑轨观感）
            if i == 0:
                self._radii = "border-top-left-radius: {r}px; border-bottom-left-radius: {r}px;"
            elif i == len(self.SEGMENTS) - 1:
                self._radii = "border-top-right-radius: {r}px; border-bottom-right-radius: {r}px;"
            else:
                self._radii = ""
            b._radii_tmpl = self._radii
        self.set_mode("stop")

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """程序化设置当前档（不发 modeChanged）。mode 可为 'loading'（无勾选段，
        仅停止可点——加载中取消加载的唯一出口）。"""
        if mode not in self._VALID:
            raise ValueError(f"未知档位: {mode}")
        self._mode = mode
        loading = mode == "loading"
        self._buttons["convert"].setText("载入" if loading else "变声")
        for m, b in self._buttons.items():
            checked = m == mode
            b.setChecked(checked)
            b.setEnabled(not loading or m == "stop")
            b.setStyleSheet(self._segment_qss(
                self._COLORS[m] if checked else None,
                b._radii_tmpl,
            ))

    @staticmethod
    def _segment_qss(fill: str | None, radii_tmpl: str) -> str:
        """单段 QSS：fill 非空 = 当前档（实心状态色白字）；None = 非当前档
        （暗底灰字、中性描边）。含 hover/disabled。"""
        radii = radii_tmpl.format(r=Layout.RADIUS_NORMAL) if radii_tmpl else ""
        checked = "" if fill is None else f"""
            QPushButton:checked {{
                background: {fill};
                color: white;
                border-color: {fill};
            }}
        """
        return f"""
            QPushButton {{
                background: {Colors.SECONDARY};
                color: {Colors.TEXT_SECONDARY};
                font-weight: bold;
                padding: 5px 8px;
                border: 1px solid {Colors.BORDER};
                {radii}
            }}
            QPushButton:hover {{
                background: {Colors.SECONDARY_HOVER};
            }}
            QPushButton:disabled {{
                color: {Colors.MUTED_TEXT};
            }}
            {checked}
        """

    def _on_clicked(self, mode: str) -> None:
        if mode == self._mode:
            # checkable 按钮点当前档会取消勾选——恢复之，无业务副作用
            self._buttons[mode].setChecked(True)
            return
        self.set_mode(mode)
        self.modeChanged.emit(mode)
