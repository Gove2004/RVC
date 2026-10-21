"""滑动条控件 — 浮点滑杆、滑杆行辅助、双滑块范围选择器。"""
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QMouseEvent, QPaintEvent, QPainter, QColor, QPen
from PySide6.QtWidgets import QSlider, QLabel, QWidget, QSizePolicy, QHBoxLayout

from gui.styles import Layout
from gui.styles.colors import Colors


class DoubleSlider(QSlider):
    """支持浮点值的滑动条，基于步长动态编码。

    对外暴露物理值（float），内部用整数编码：
        整数值 = round(物理值 / step)
        物理值 = 整数值 × step

    拖动时的最小变化量即 step；存档直接存物理值，不需要编码转换。
    """

    def __init__(self, min_val: float, max_val: float, step: float,
                 parent: QWidget | None = None,
                 value_label: QLabel | None = None,
                 label_fmt: str = ".2f", label_unit: str = "") -> None:
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

    def value(self) -> float:  # 刻意放宽：对外是物理值(float)
        """返回物理值（float）。"""
        return super().value() * self._step

    def setValue(self, value: float) -> None:
        """设置物理值（float）。"""
        super().setValue(round(float(value) / self._step))

    def minimum(self) -> float:
        return self._int_min * self._step

    def maximum(self) -> float:
        return self._int_max * self._step

    def _fmt(self, v: float) -> str:
        return f"{v:{self._label_fmt}}{self._label_unit}"

    def _update_label(self) -> None:
        """刷新值标签：拖动经 valueChanged 自动触发；程序化 setValue 不触发
        valueChanged 时（如控件未显示）由调用方手动调用。"""
        if self._value_label is not None:
            self._value_label.setText(self._fmt(self.value()))


def _create_slider_row(mn: float, mx: float, st: float, dv: float,
                       fmt: str = ".2f", unit: str = "",
                       label_w: int = 80) -> tuple[DoubleSlider, QLabel]:
    """创建「滑杆 + 自动格式化值标签」，返回 (slider, label)。

    基于 DoubleSlider，对外直接暴露物理值（float），拖动最小变化量 = step。
    fmt/unit 控制标签显示，label_w 控制标签宽度（默认 80px 定宽）。
    控件由调用点具名挂载，helper 不做 setattr 动态挂窗口。
    """
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    s = DoubleSlider(mn, mx, st, value_label=lbl, label_fmt=fmt, label_unit=unit)
    s.setFixedHeight(Layout.SLIDER_HEIGHT)
    s.setValue(dv)
    s._update_label()
    s.valueChanged.connect(s._update_label)
    return s, lbl


class RangeSlider(QWidget):
    """双滑块范围选择器 — 一个控件同时控制下限和上限。

    内部直接存物理值（float），基于步长量化，和 DoubleSlider 统一。
    low()/high() 返回物理值（float）。拖动时的最小变化量 = step。
    """
    rangeChanged = Signal(float, float)  # low, high（物理值）

    def __init__(self, min_val: float, max_val: float, step: float,
                 low_val: float, high_val: float,
                 fmt: str = ".0f", unit: str = "Hz",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min = float(min_val)
        self._max = float(max_val)
        self._step = float(step)
        self._low = float(low_val)
        self._high = float(high_val)
        self._fmt = fmt
        self._unit = unit
        self._dragging: str | None = None  # 'low' or 'high'
        self.setFixedHeight(Layout.SLIDER_HEIGHT)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_tooltip()

    def setRange(self, low_val: float, high_val: float) -> None:
        """设置当前范围（物理值），并发射 rangeChanged 信号。"""
        self._low = max(self._min, float(low_val))
        self._high = min(self._max, float(high_val))
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        self.update()
        self._update_tooltip()
        self.rangeChanged.emit(self._low, self._high)

    def low(self) -> float:
        return self._low

    def high(self) -> float:
        return self._high

    def _update_tooltip(self) -> None:
        self.setToolTip(f"{self._low:{self._fmt}}{self._unit} - "
                        f"{self._high:{self._fmt}}{self._unit}")

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        track_h = 4
        track_y = h // 2 - track_h // 2  # 轨道居中
        margin = 6
        track_w = w - 2 * margin

        # 背景轨道
        painter.setPen(Qt.PenStyle.NoPen)
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

    def _value_to_x(self, value: float) -> int:
        ratio = (value - self._min) / (self._max - self._min)
        return int(6 + ratio * (self.width() - 12))

    def _x_to_value(self, x: int) -> float:
        ratio = max(0.0, min(1.0, (x - 6) / (self.width() - 12)))
        value = self._min + ratio * (self._max - self._min)
        return round(value / self._step) * self._step

    def _event_x(self, event: QMouseEvent | None) -> int:
        """鼠标事件的 x 坐标（QMouseEvent 的 position 是 QPointF，取整到像素）。"""
        if event is None:
            return 0
        return event.position().toPoint().x()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        x = self._event_x(event)
        low_x = self._value_to_x(self._low)
        high_x = self._value_to_x(self._high)
        if abs(x - low_x) < 12:
            self._dragging = 'low'
        elif abs(x - high_x) < 12:
            self._dragging = 'high'
        else:
            # 点击轨道时，移动最近的滑块
            if abs(x - low_x) < abs(x - high_x):
                self._dragging = 'low'
                self._update_low(x)
            else:
                self._dragging = 'high'
                self._update_high(x)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        x = self._event_x(event)
        if self._dragging == 'low':
            self._update_low(x)
        elif self._dragging == 'high':
            self._update_high(x)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        self._dragging = None

    def _update_low(self, x: int) -> None:
        new_low = self._x_to_value(x)
        new_low = min(new_low, self._high - self._step)
        new_low = max(new_low, self._min)
        if new_low != self._low:
            self._low = new_low
            self.update()
            self._update_tooltip()
            self.rangeChanged.emit(self._low, self._high)

    def _update_high(self, x: int) -> None:
        new_high = self._x_to_value(x)
        new_high = max(new_high, self._low + self._step)
        new_high = min(new_high, self._max)
        if new_high != self._high:
            self._high = new_high
            self.update()
            self._update_tooltip()
            self.rangeChanged.emit(self._low, self._high)
