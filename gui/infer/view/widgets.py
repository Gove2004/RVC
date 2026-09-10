"""推理 GUI 通用组件 — 滑动条辅助、加载线程、双滑块范围控件"""
from PySide6.QtWidgets import QSlider, QLabel, QWidget, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from rvc.core.config import HUBERT_DEFAULT

__all__ = ["LoadThread", "_sl", "_slrow", "_sl_value_as_float", "RangeSlider"]


def _sl(mn, mx, st, dv):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


def _slrow(win, attr, mn, mx, st, dv, fmt=".2f", unit="", label_w=80):
    """创建「滑杆 + 自动格式化值标签」并挂到 win.<attr> / win.<attr>_label。

    参数为**物理值**（工厂内部按 ×100 编码到 QSlider；运行时 QSlider.value()/100
    即物理值，与 param_binding 的 X100 读写兼容）。fmt/unit 控制标签显示。
    返回 slider。一处样板替换原先「建滑块 + 建 label + connect 格式化」三行。
    """
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(int(mn * 100), int(mx * 100))
    s.setSingleStep(int(st * 100))
    s.setValue(int(dv * 100))
    lbl = QLabel()
    lbl.setMinimumWidth(label_w)

    def _fmt(v):
        return f"{v / 100:{fmt}}{unit}"

    lbl.setText(_fmt(s.value()))
    s.valueChanged.connect(lambda v: lbl.setText(_fmt(v)))
    setattr(win, attr, s)
    # 约定：滑块属性 xxx_slider → 值标签属性 xxx_label
    label_attr = attr[:-7] + "_label" if attr.endswith("_slider") else attr + "_label"
    setattr(win, label_attr, lbl)
    return s


def _sl_value_as_float(slider: QSlider, divisor: float = 100.0) -> float:
    return slider.value() / divisor


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

    采用 X100 编码（和 _slrow 兼容）：内部值为物理值×100，
    low()/high() 返回物理值（float）。
    """
    rangeChanged = Signal(float, float)  # low, high（物理值）

    def __init__(self, min_val, max_val, step, low_val, high_val,
                 fmt=".0f", unit="Hz", parent=None):
        super().__init__(parent)
        self._min = int(min_val * 100)
        self._max = int(max_val * 100)
        self._step = int(step * 100)
        self._low = int(low_val * 100)
        self._high = int(high_val * 100)
        self._fmt = fmt
        self._unit = unit
        self._dragging = None  # 'low' or 'high'
        self.setMinimumHeight(18)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_tooltip()

    def setRange(self, low_val, high_val):
        """设置当前范围（物理值）。"""
        self._low = max(self._min, int(low_val * 100))
        self._high = min(self._max, int(high_val * 100))
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        self.update()
        self._update_tooltip()

    def low(self):
        return self._low / 100.0

    def high(self):
        return self._high / 100.0

    def _update_tooltip(self):
        self.setToolTip(f"{self._low/100:{self._fmt}}{self._unit} - "
                        f"{self._high/100:{self._fmt}}{self._unit}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        track_h = 4
        track_y = h // 2 - track_h // 2  # 轨道居中
        margin = 8
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
        return int(round(value / self._step) * self._step)

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
            self.rangeChanged.emit(self._low / 100.0, self._high / 100.0)

    def _update_high(self, x):
        new_high = self._x_to_value(x)
        new_high = max(new_high, self._low + self._step)
        new_high = min(new_high, self._max)
        if new_high != self._high:
            self._high = new_high
            self.update()
            self._update_tooltip()
            self.rangeChanged.emit(self._low / 100.0, self._high / 100.0)
