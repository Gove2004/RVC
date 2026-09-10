"""推理 GUI 通用组件 — 滑动条辅助、加载线程"""
from PySide6.QtWidgets import QSlider, QLabel
from PySide6.QtCore import Qt, QThread, Signal

from rvc.core.config import HUBERT_DEFAULT

__all__ = ["LoadThread", "_sl", "_slrow", "_sl_value_as_float"]


def _sl(mn, mx, st, dv):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


def _slrow(win, attr, mn, mx, st, dv, fmt=".2f", unit="", label_w=35):
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
