"""推理 GUI 通用组件 — 模型卡片、模型列表数据、加载线程"""
import os

from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal

from gui.configs import load_config, save_config
from gui.styles import ButtonStyles, LabelStyles, CardStyles, Layout, Colors

__all__ = ["ModelCard", "ModelListData", "LoadThread", "_sl", "_sl_value_as_float"]


def _sl(mn, mx, st, dv):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


def _sl_value_as_float(slider: QSlider, divisor: float = 100.0) -> float:
    return slider.value() / divisor


class ModelListData:
    @staticmethod
    def load():
        cfg = load_config()
        return cfg.get("models", [])

    @staticmethod
    def save(models):
        cfg = load_config()
        cfg["models"] = models
        save_config(cfg)


class ModelCard(QFrame):
    """模型卡片：始终展开，顶部一行 [使用] [模型名居中] [删除]"""

    load_requested = Signal(str, str, str, float, float, float, float)  # 7个参数，移除protect

    def __init__(self, name="", pth="", idx="", pitch=0,
                 index_rate=0.0, gender=0.0, parent=None):
        super().__init__(parent)
        self._build(name, pth, idx, pitch, index_rate, gender)

    def _build(self, name, pth, idx, pitch, index_rate, gender):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(4)

        # ── 顶部栏 ──
        hdr = QHBoxLayout()
        self._btn_use = QPushButton("使用")
        self._btn_use.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self._btn_use.setStyleSheet(ButtonStyles.primary())
        self._btn_use.clicked.connect(self._on_load)
        hdr.addWidget(self._btn_use)
        hdr.addSpacing(8)

        self._name_label = QLabel(name or os.path.splitext(os.path.basename(pth))[0])
        self._name_label.setStyleSheet(LabelStyles.bold())
        hdr.addWidget(self._name_label, 1)
        hdr.addSpacing(8)

        self._del = QPushButton("删除")
        self._del.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self._del.setStyleSheet(ButtonStyles.danger())
        hdr.addWidget(self._del)
        root.addLayout(hdr)

        # ── 参数区 ──
        body = QWidget(); bl = QGridLayout(body); bl.setContentsMargins(24, 2, 6, 4); bl.setSpacing(3)
        r = 0

        def add_path_row(label_text, line_edit, filter_text, row):
            bl.addWidget(QLabel(label_text), row, 0)
            btn = QPushButton("…"); btn.setFixedSize(Layout.BTN_WIDTH_ICON, Layout.BTN_HEIGHT_SMALL)
            btn.setStyleSheet(ButtonStyles.small())
            container = QWidget()
            cl = QHBoxLayout(container); cl.setContentsMargins(0, 0, 0, 0)
            cl.addWidget(line_edit, 1); cl.addWidget(btn)
            bl.addWidget(container, row, 1, 1, 2)
            return btn

        self.pth_edit = QLineEdit(pth); self.pth_edit.setMinimumHeight(24)
        _pbtn = add_path_row("模型路径", self.pth_edit, "模型 (*.pth)", r)
        _pbtn.clicked.connect(lambda: self._browse(self.pth_edit, "模型 (*.pth)"))
        r += 1

        self.idx_edit = QLineEdit(idx); self.idx_edit.setMinimumHeight(24)
        _ibtn = add_path_row("索引路径", self.idx_edit, "索引 (*.index)", r)
        _ibtn.clicked.connect(lambda: self._browse(self.idx_edit, "索引 (*.index)"))
        r += 1

        self.pitch_slider = _sl(-16, 16, 1, pitch); self.pitch_label = QLabel(str(pitch))
        self.pitch_slider.valueChanged.connect(lambda v: self.pitch_label.setText(str(v)))
        bl.addWidget(QLabel("音调大小"), r, 0); bl.addWidget(self.pitch_slider, r, 1); bl.addWidget(self.pitch_label, r, 2); r += 1

        gender_slider_val = int(((gender + 2.5) / 5.0) * 100)
        self.gender_slider = _sl(0, 100, 1, gender_slider_val); self.gender_label = QLabel(f"{gender:+.2f}")
        self.gender_slider.valueChanged.connect(lambda v: self.gender_label.setText(f"{(v / 100 * 5 - 2.5):+.2f}"))
        bl.addWidget(QLabel("性别因子"), r, 0); bl.addWidget(self.gender_slider, r, 1); bl.addWidget(self.gender_label, r, 2); r += 1

        self.index_rate_slider = _sl(0, 100, 1, int(index_rate * 100)); self.index_rate_label = QLabel(f"{index_rate:.2f}")
        self.index_rate_slider.valueChanged.connect(lambda v: self.index_rate_label.setText(f"{v / 100:.2f}"))
        bl.addWidget(QLabel("索引占比"), r, 0); bl.addWidget(self.index_rate_slider, r, 1); bl.addWidget(self.index_rate_label, r, 2); r += 1

        root.addWidget(body)
        self.setStyleSheet(f"ModelCard{{{CardStyles.default()}}}")

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path:
            tgt.setText(path)

    def _on_load(self):
        self.load_requested.emit(
            self._name_label.text(), self.pth_edit.text().strip(), self.idx_edit.text().strip(),
            self.pitch_slider.value(), _sl_value_as_float(self.index_rate_slider), 0.0,
            _sl_value_as_float(self.gender_slider),
        )

    def get_data(self):
        return {
            "name": self._name_label.text(),
            "pth": self.pth_edit.text().strip(),
            "idx": self.idx_edit.text().strip(),
            "pitch": self.pitch_slider.value(),
            "index_rate": _sl_value_as_float(self.index_rate_slider),
            "gender": (self.gender_slider.value() / 100 * 5 - 2.5),
        }

    def set_active(self, active):
        if active:
            self._btn_use.setText("使用中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet(ButtonStyles.primary())
            self.setStyleSheet(f"ModelCard{{{CardStyles.active('success')}}}")
            self._name_label.setStyleSheet(LabelStyles.status("success"))
        else:
            self._btn_use.setText("使用")
            self._btn_use.setEnabled(True)
            self._btn_use.setStyleSheet(ButtonStyles.primary())
            self.setStyleSheet(f"ModelCard{{{CardStyles.default()}}}")
            self._name_label.setStyleSheet(LabelStyles.bold())

    def set_loading(self, loading):
        if loading:
            self._btn_use.setText("加载中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet(ButtonStyles.secondary())
            self.setStyleSheet(f"ModelCard{{{CardStyles.active('info')}}}")
            self._name_label.setStyleSheet(LabelStyles.status("info"))


class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, engine, pth, idx, idx_rate):
        super().__init__()
        self.engine = engine; self.pth = pth; self.idx = idx; self.rate = idx_rate
        self._stop_requested = False
    def request_stop(self):
        self._stop_requested = True
    def is_stopping(self):
        return self._stop_requested
    def run(self):
        try:
            if not self.is_stopping():
                self.ok.emit(self.engine.load_model(self.pth, self.idx, self.rate, True))
        except Exception as e:
            self.err.emit(str(e))
