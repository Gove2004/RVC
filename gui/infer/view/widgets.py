"""推理 GUI 通用组件 — 滑动条辅助、加载线程、双滑块范围控件"""
from PySide6.QtWidgets import QSlider, QLabel, QWidget, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from rvc.core.config import HUBERT_DEFAULT

__all__ = ["LoadThread", "_sl", "_slrow", "_sl_value_as_float", "DoubleSlider", "RangeSlider"]


def _sl(mn, mx, st, dv):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


class DoubleSlider(QSlider):
    """支持浮点值的滑动条，基于步长动态编码。

    对外暴露物理值（float），内部用整数编码：
        整数值 = round(物理值 / step)
        物理值 = 整数值 × step

    这样拖动时的最小变化量就是 step，而不是固定的 0.01（X100 编码的问题）。
    存档直接存物理值，不需要 X100 转换。
    """

    def __init__(self, min_val, max_val, step, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._step = float(step)
        self._int_min = round(min_val / self._step)
        self._int_max = round(max_val / self._step)
        super().setMinimum(self._int_min)
        super().setMaximum(self._int_max)
        super().setSingleStep(1)

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


def _slrow(win, attr, mn, mx, st, dv, fmt=".2f", unit="", label_w=80):
    """创建「滑杆 + 自动格式化值标签」并挂到 win.<attr> / win.<attr>_label。

    使用 DoubleSlider，基于步长动态编码，对外直接暴露物理值（float）。
    存档直接存物理值，不需要 X100 转换。拖动时的最小变化量 = step。
    fmt/unit 控制标签显示。
    返回 slider。
    """
    s = DoubleSlider(mn, mx, st)
    s.setFixedHeight(18)
    s.setValue(dv)
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignCenter)

    def _fmt(v):
        return f"{v:{fmt}}{unit}"

    lbl.setText(_fmt(s.value()))
    s.valueChanged.connect(lambda _v: lbl.setText(_fmt(s.value())))
    setattr(win, attr, s)
    # 约定：滑块属性 xxx_slider → 值标签属性 xxx_label
    label_attr = attr[:-7] + "_label" if attr.endswith("_slider") else attr + "_label"
    setattr(win, label_attr, lbl)
    return s


def _sl_value_as_float(slider) -> float:
    """从滑动条读取物理值（DoubleSlider.value() 已返回 float）。"""
    return float(slider.value())


class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, engine, pth, hubert=HUBERT_DEFAULT):
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
        self.setFixedHeight(18)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_tooltip()

    def setRange(self, low_val, high_val):
        """设置当前范围（物理值）。"""
        self._low = max(self._min, float(low_val))
        self._high = min(self._max, float(high_val))
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        self.update()
        self._update_tooltip()

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
        painter.setBrush(QColor("#3a3a3a"))
        painter.drawRoundedRect(QRect(margin, track_y, track_w, track_h), 2, 2)

        # 选中范围
        low_x = self._value_to_x(self._low)
        high_x = self._value_to_x(self._high)
        painter.setBrush(QColor("#0078D4"))
        painter.drawRoundedRect(QRect(low_x, track_y, high_x - low_x, track_h), 2, 2)

        # 滑块（方形，和 QSlider 样式一致）
        handle_size = 12
        for x in (low_x, high_x):
            painter.setBrush(QColor("#0078D4"))
            painter.setPen(QPen(QColor("#1a1a1a"), 1))
            handle_rect = QRect(x - handle_size // 2,
                                track_y + track_h // 2 - handle_size // 2,
                                handle_size, handle_size)
            painter.drawRoundedRect(handle_rect, 2, 2)

    def _value_to_x(self, value):
        ratio = (value - self._min) / (self._max - self._min)
        return int(8 + ratio * (self.width() - 16))

    def _x_to_value(self, x):
        ratio = max(0.0, min(1.0, (x - 8) / (self.width() - 16)))
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
