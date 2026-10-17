"""推理 GUI 通用组件 — 滑动条辅助、加载线程、双滑块范围控件、路径选择按钮"""
from pathlib import Path

from PySide6.QtWidgets import (
    QSlider, QLabel, QWidget, QSizePolicy, QPushButton, QFileDialog,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QPen

from gui.styles import ButtonStyles, Layout
from gui.styles.colors import Colors

__all__ = [
    "LoadThread", "_create_slider_row", "DoubleSlider", "RangeSlider",
    "PathPickerButton", "ElidedLabel", "ModeToggle",
]


class PathPickerButton(QPushButton):
    """路径选择按钮 — 自带文件对话框的路径控件。

    按钮文本显示所选文件名（未选时显示 placeholder），tooltip 显示完整路径；
    路径由控件自身持有（`path()` / `set_path()`），调用方不再维护平行的字符串字段。

    取代「只读文本框 + 浏览按钮」两件套：音色页签的模型路径、离线页签的输入文件、
    融合/详情页签的模型文件统一使用本控件，交互一致（点一下即弹选择框）。
    """

    pathChanged = Signal(str)

    def __init__(self, parent=None, *, title="选择文件",
                 filter_str="PyTorch (*.pth)", placeholder="选择文件...",
                 base_dir=None, tooltip=None):
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._title = title
        self._filter = filter_str
        # base_dir 允许传 callable：起始目录可能来自惰性构造的 controller
        self._base_dir = base_dir
        self._path = ""
        self.setMinimumHeight(Layout.BTN_HEIGHT_NORMAL)
        self.setStyleSheet(ButtonStyles.path_button())
        self.setToolTip(tooltip or self._idle_tooltip())
        self.clicked.connect(self._pick)

    def _idle_tooltip(self) -> str:
        return f"点击{self._placeholder.rstrip('.')}"

    def _pick(self) -> None:
        base = self._base_dir() if callable(self._base_dir) else (self._base_dir or "")
        path, _ = QFileDialog.getOpenFileName(self, self._title, str(base), self._filter)
        if path:
            self.set_path(path)

    def path(self) -> str:
        """当前选中的完整路径（未选择时为空串）。"""
        return self._path

    def set_path(self, path: str, *, emit: bool = True) -> None:
        """程序化设置路径（配置回填 / 清空）；emit=False 时不触发 pathChanged。"""
        self._path = path or ""
        if self._path:
            self.setText(Path(self._path).name)
            self.setToolTip(self._path)
        else:
            self.setText(self._placeholder)
            self.setToolTip(self._idle_tooltip())
        if emit:
            self.pathChanged.emit(self._path)


class ElidedLabel(QLabel):
    """单行只读文本标签 — 超宽时按中间省略号截断，完整内容放在 tooltip。

    用于展示「由其它输入派生、用户不可编辑」的路径/结果文本：既不使用输入框
    （不可编辑，视觉上也不是输入控件），又不会因文本过长把所在页签撑宽
    （sizeHint 被 Ignored，宽度完全跟随布局分配）。
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def text(self) -> str:
        """完整文本（未截断）。"""
        return self._full

    def setText(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._relayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        fm = self.fontMetrics()
        width = max(0, self.width() - 2)
        super().setText(fm.elidedText(self._full, Qt.TextElideMode.ElideMiddle, width))


class DoubleSlider(QSlider):
    """支持浮点值的滑动条，基于步长动态编码。

    对外暴露物理值（float），内部用整数编码：
        整数值 = round(物理值 / step)
        物理值 = 整数值 × step

    拖动时的最小变化量即 step；存档直接存物理值，不需要编码转换。
    """

    def __init__(self, min_val, max_val, step, parent=None,
                 value_label=None, label_fmt=".2f", label_unit=""):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._step = float(step)
        self._int_min = round(min_val / self._step)
        self._int_max = round(max_val / self._step)
        super().setMinimum(self._int_min)
        super().setMaximum(self._int_max)
        super().setSingleStep(1)
        # 值标签与格式化参数
        self._value_label = value_label
        self._label_fmt = label_fmt
        self._label_unit = label_unit

    def value(self):
        """返回物理值（float）。"""
        return super().value() * self._step

    def setValue(self, value):
        """设置物理值（float）。"""
        super().setValue(round(float(value) / self._step))

    def minimum(self):
        return self._int_min * self._step

    def maximum(self):
        return self._int_max * self._step

    def _fmt(self, v):
        return f"{v:{self._label_fmt}}{self._label_unit}"

    def _update_label(self) -> None:
        """刷新值标签：拖动经 valueChanged 自动触发；程序化 setValue 不触发
        valueChanged 时（如控件未显示）由调用方手动调用。"""
        self._value_label.setText(self._fmt(self.value()))


def _create_slider_row(mn, mx, st, dv, fmt=".2f", unit="", label_w=80):
    """创建「滑杆 + 自动格式化值标签」，返回 (slider, label)。

    基于 DoubleSlider，对外直接暴露物理值（float），拖动最小变化量 = step。
    fmt/unit 控制标签显示，label_w 控制标签宽度（默认 80px 定宽）。
    控件由调用点具名挂载，helper 不做 setattr 动态挂窗口。
    """
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignCenter)
    s = DoubleSlider(mn, mx, st, value_label=lbl, label_fmt=fmt, label_unit=unit)
    s.setFixedHeight(Layout.SLIDER_HEIGHT)
    s.setValue(dv)
    s._update_label()
    s.valueChanged.connect(s._update_label)
    return s, lbl


class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, engine, pth, hubert):
        super().__init__()
        self.engine = engine; self.pth = pth; self.hubert = hubert
        self._stop_requested = False
    def request_stop(self):
        self._stop_requested = True
    def is_stopping(self):
        return self._stop_requested
    def run(self):
        try:
            if not self.is_stopping():
                self.ok.emit(self.engine.load_model(
                    self.pth, True, self.hubert))
        except Exception as e:
            self.err.emit(str(e))


class RangeSlider(QWidget):
    """双滑块范围选择器 — 一个控件同时控制下限和上限。

    内部直接存物理值（float），基于步长量化，和 DoubleSlider 统一。
    low()/high() 返回物理值（float）。拖动时的最小变化量 = step。
    """
    rangeChanged = Signal(float, float)  # low, high（物理值）

    def __init__(self, min_val, max_val, step, low_val, high_val,
                 fmt=".0f", unit="Hz", parent=None):
        super().__init__(parent)
        self._min = float(min_val)
        self._max = float(max_val)
        self._step = float(step)
        self._low = float(low_val)
        self._high = float(high_val)
        self._fmt = fmt
        self._unit = unit
        self._dragging = None  # 'low' or 'high'
        self.setFixedHeight(Layout.SLIDER_HEIGHT)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_tooltip()

    def setRange(self, low_val, high_val):
        """设置当前范围（物理值），并发射 rangeChanged 信号。"""
        self._low = max(self._min, float(low_val))
        self._high = min(self._max, float(high_val))
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        self.update()
        self._update_tooltip()
        self.rangeChanged.emit(self._low, self._high)

    def low(self):
        return self._low

    def high(self):
        return self._high

    def _update_tooltip(self):
        self.setToolTip(f"{self._low:{self._fmt}}{self._unit} - "
                        f"{self._high:{self._fmt}}{self._unit}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        track_h = 4
        track_y = h // 2 - track_h // 2  # 轨道居中
        margin = 6
        track_w = w - 2 * margin

        # 背景轨道
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Colors.BORDER))
        painter.drawRoundedRect(QRect(margin, track_y, track_w, track_h), 2, 2)

        # 选中范围
        low_x = self._value_to_x(self._low)
        high_x = self._value_to_x(self._high)
        painter.setBrush(QColor(Colors.PRIMARY))
        painter.drawRoundedRect(QRect(low_x, track_y, high_x - low_x, track_h), 2, 2)

        # 滑块（方形，和 QSlider 样式一致）
        handle_size = 12
        for x in (low_x, high_x):
            painter.setBrush(QColor(Colors.PRIMARY))
            painter.setPen(QPen(QColor(Colors.INPUT_BG), 1))
            handle_rect = QRect(x - handle_size // 2,
                                track_y + track_h // 2 - handle_size // 2,
                                handle_size, handle_size)
            painter.drawRoundedRect(handle_rect, 2, 2)

    def _value_to_x(self, value):
        ratio = (value - self._min) / (self._max - self._min)
        return int(6 + ratio * (self.width() - 12))

    def _x_to_value(self, x):
        ratio = max(0.0, min(1.0, (x - 6) / (self.width() - 12)))
        value = self._min + ratio * (self._max - self._min)
        return round(value / self._step) * self._step

    def mousePressEvent(self, event):
        low_x = self._value_to_x(self._low)
        high_x = self._value_to_x(self._high)
        if abs(event.x() - low_x) < 12:
            self._dragging = 'low'
        elif abs(event.x() - high_x) < 12:
            self._dragging = 'high'
        else:
            # 点击轨道时，移动最近的滑块
            if abs(event.x() - low_x) < abs(event.x() - high_x):
                self._dragging = 'low'
                self._update_low(event.x())
            else:
                self._dragging = 'high'
                self._update_high(event.x())

    def mouseMoveEvent(self, event):
        if self._dragging == 'low':
            self._update_low(event.x())
        elif self._dragging == 'high':
            self._update_high(event.x())

    def mouseReleaseEvent(self, event):
        self._dragging = None

    def _update_low(self, x):
        new_low = self._x_to_value(x)
        new_low = min(new_low, self._high - self._step)
        new_low = max(new_low, self._min)
        if new_low != self._low:
            self._low = new_low
            self.update()
            self._update_tooltip()
            self.rangeChanged.emit(self._low, self._high)

    def _update_high(self, x):
        new_high = self._x_to_value(x)
        new_high = max(new_high, self._low + self._step)
        new_high = min(new_high, self._max)
        if new_high != self._high:
            self._high = new_high
            self.update()
            self._update_tooltip()
            self.rangeChanged.emit(self._low, self._high)


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
        self._buttons["convert"].setText("加载中" if loading else "变声")
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
