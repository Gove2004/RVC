"""推理 GUI 通用组件 — 模型卡片、模型列表数据、加载线程"""
import os

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal

from gui.configs import load_config, save_config
from gui.styles import ButtonStyles, LabelStyles, CardStyles, Layout, Colors
from gui.infer.param_binding import formant_to_gender, gender_to_formant
from rvc.inference.params import HUBERT_DEFAULT

__all__ = ["ModelCard", "ModelListData", "LoadThread", "_sl", "_slrow", "_sl_value_as_float"]


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

    # name, pth, pitch, gender, hubert
    load_requested = Signal(str, str, int, float, str)

    def __init__(self, name="", pth="", pitch=12,
                 gender=0.0, hubert=HUBERT_DEFAULT, parent=None):
        super().__init__(parent)
        self._build(name, pth, pitch, gender, hubert)

    def _build(self, name, pth, pitch, gender, hubert):
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

        self.pitch_slider = _sl(-16, 16, 1, pitch); self.pitch_label = QLabel(str(pitch))
        self.pitch_slider.valueChanged.connect(lambda v: self.pitch_label.setText(str(v)))
        bl.addWidget(QLabel("音调大小"), r, 0); bl.addWidget(self.pitch_slider, r, 1); bl.addWidget(self.pitch_label, r, 2); r += 1

        # 滑杆 [0,1] ↔ formant shift [-2.5,+2.5]，换算唯一来源在 param_binding
        # （gender_to_formant / formant_to_gender），此处禁止内联手写公式
        gender_slider_val = int(round(formant_to_gender(gender) * 100))
        self.gender_slider = _sl(0, 100, 1, gender_slider_val); self.gender_label = QLabel(f"{gender:+.2f}")
        self.gender_slider.valueChanged.connect(lambda v: self.gender_label.setText(f"{gender_to_formant(v / 100):+.2f}"))
        bl.addWidget(QLabel("性别因子"), r, 0); bl.addWidget(self.gender_slider, r, 1); bl.addWidget(self.gender_label, r, 2); r += 1

        # HuBERT 特征器：base（原始 hubert_base）/ chinese（腾讯中文 hubert）。
        # 硬约束：训练与推理必须用同一特征器——本模型训练时用的哪个，这里就要选哪个。
        self.hubert_combo = QComboBox()
        self.hubert_combo.addItems(["base", "chinese"])
        self.hubert_combo.setMinimumHeight(24)
        bi = self.hubert_combo.findText(hubert)
        if bi >= 0:
            self.hubert_combo.setCurrentIndex(bi)
        self.hubert_combo.setToolTip("此模型训练时用的特征器（base=原版 hubert_base，chinese=腾讯中文 hubert）。训练与推理必须一致。")
        bl.addWidget(QLabel("特征器"), r, 0); bl.addWidget(self.hubert_combo, r, 1, 1, 2); r += 1

        root.addWidget(body)
        self.setStyleSheet(f"ModelCard{{{CardStyles.default()}}}")

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path:
            tgt.setText(path)

    def _on_load(self):
        self.load_requested.emit(
            self._name_label.text(), self.pth_edit.text().strip(),
            self.pitch_slider.value(),
            _sl_value_as_float(self.gender_slider),
            self.hubert_combo.currentText(),
        )

    def get_data(self):
        return {
            "name": self._name_label.text(),
            "pth": self.pth_edit.text().strip(),
            "pitch": self.pitch_slider.value(),
            "gender": gender_to_formant(self.gender_slider.value() / 100),
            "hubert": self.hubert_combo.currentText(),
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
