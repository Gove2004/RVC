"""推理 GUI 通用组件 — 模型卡片、模型列表数据、加载线程"""
import os

from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal

from gui.configs import load_state_json, save_state_json
from gui.styles import ButtonStyles, LabelStyles, CardStyles, Layout

__all__ = ["ModelCard", "ModelListData", "LoadThread", "_sl", "_sl_value_as_float"]


def _sl(mn, mx, st, dv):
    """创建水平滑块的快捷函数"""
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


def _sl_value_as_float(slider: QSlider, divisor: float = 100.0) -> float:
    """将滑块整数值转换为浮点数。

    Args:
        slider: QSlider 实例
        divisor: 除数（默认 100.0，即将 0-100 映射到 0.0-1.0）

    Returns:
        float: 转换后的浮点值
    """
    return slider.value() / divisor


class ModelListData:
    """管理模型列表的持久化"""

    @staticmethod
    def load():
        return load_state_json("models", {"models": []}).get("models", [])

    @staticmethod
    def save(models):
        save_state_json("models", {"models": models})


class ModelCard(QFrame):
    """模型卡片: 使用按钮选中, 展开按钮显示参数"""

    load_requested = Signal(str, str, str, float, float, float, float, float)

    def __init__(self, name="", pth="", idx="", pitch=0,
                 index_rate=0.0, rms_mix=0.0, gender=50, protect=50, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._build(name, pth, idx, pitch, index_rate, rms_mix, gender, protect)
        self._body.setVisible(False)

    def _build(self, name, pth, idx, pitch, index_rate, rms_mix, gender, protect):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # 头部: 名称 + 使用按钮 + 展开按钮
        hdr = QWidget()
        hl = QHBoxLayout(hdr); hl.setContentsMargins(6,5,6,5)
        self._name = QLabel(name or os.path.splitext(os.path.basename(pth))[0])
        self._name.setStyleSheet(LabelStyles.bold())
        self._btn_use = QPushButton("使用")
        self._btn_use.setFixedWidth(40)
        self._btn_use.setStyleSheet(ButtonStyles.small("secondary"))
        self._btn_use.clicked.connect(self._on_load)
        self._btn_expand = QPushButton("展开")
        self._btn_expand.setFixedWidth(36)
        self._btn_expand.clicked.connect(self._toggle)
        hl.addWidget(self._name, 1)
        hl.addWidget(self._btn_use)
        hl.addWidget(self._btn_expand)
        root.addWidget(hdr)

        # 内容 (展开后显示)
        self._body = QWidget()
        bl = QGridLayout(self._body); bl.setContentsMargins(24,2,6,4); bl.setSpacing(2)
        r = 0
        bl.addWidget(QLabel("模型"), r, 0)
        self.pth_edit = QLineEdit(pth); bl.addWidget(self.pth_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(Layout.BTN_WIDTH_ICON, Layout.BTN_HEIGHT_SMALL)
        b.setStyleSheet(ButtonStyles.small())
        b.clicked.connect(lambda: self._browse(self.pth_edit, "模型 (*.pth)")); bl.addWidget(b, r, 2); r+=1
        bl.addWidget(QLabel("索引"), r, 0)
        self.idx_edit = QLineEdit(idx); bl.addWidget(self.idx_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(Layout.BTN_WIDTH_ICON, Layout.BTN_HEIGHT_SMALL)
        b.setStyleSheet(ButtonStyles.small())
        b.clicked.connect(lambda: self._browse(self.idx_edit, "索引 (*.index)")); bl.addWidget(b, r, 2); r+=1

        def add_slider(label, slider, value_label, row):
            bl.addWidget(QLabel(label), row, 0); bl.addWidget(slider, row, 1); bl.addWidget(value_label, row, 2)

        self.pitch_slider = _sl(-16,16,1,pitch); self.pitch_label = QLabel(str(pitch))
        self.pitch_slider.valueChanged.connect(lambda v: self.pitch_label.setText(str(v)))
        add_slider("音调", self.pitch_slider, self.pitch_label, r); r+=1

        self.gender_slider = _sl(0,100,1,gender); self.gender_label = QLabel(f"{(gender/100-0.5)*4:+.2f}")
        self.gender_slider.valueChanged.connect(lambda v: self.gender_label.setText(f"{(v/100-0.5)*4:+.2f}"))
        add_slider("性别", self.gender_slider, self.gender_label, r); r+=1

        self.index_rate_slider = _sl(0,100,1,int(index_rate*100)); self.index_rate_label = QLabel(f"{index_rate:.2f}")
        self.index_rate_slider.valueChanged.connect(lambda v: self.index_rate_label.setText(f"{v/100:.2f}"))
        add_slider("索引", self.index_rate_slider, self.index_rate_label, r); r+=1

        self.rms_mix_slider = _sl(0,100,1,int(rms_mix*100)); self.rms_mix_label = QLabel(f"{rms_mix:.2f}")
        self.rms_mix_slider.valueChanged.connect(lambda v: self.rms_mix_label.setText(f"{v/100:.2f}"))
        add_slider("响度", self.rms_mix_slider, self.rms_mix_label, r); r+=1

        self.protect_slider = _sl(0,100,1,protect); self.protect_label = QLabel(f"{protect/100:.2f}")
        self.protect_slider.valueChanged.connect(lambda v: self.protect_label.setText(f"{v/100:.2f}"))
        add_slider("辅音保护", self.protect_slider, self.protect_label, r); r+=1

        self._del = QPushButton("删除此模型")
        self._del.setStyleSheet(ButtonStyles.danger())
        bl.addWidget(self._del, r, 0, 1, 3)
        root.addWidget(self._body)
        self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._btn_expand.setText("折叠" if self._expanded else "展开")

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path: tgt.setText(path)

    def _on_load(self):
        self.load_requested.emit(
            self._name.text(), self.pth_edit.text().strip(), self.idx_edit.text().strip(),
            self.pitch_slider.value(), _sl_value_as_float(self.index_rate_slider), _sl_value_as_float(self.rms_mix_slider),
            _sl_value_as_float(self.gender_slider), _sl_value_as_float(self.protect_slider),
        )

    def get_data(self):
        return {
            "name": self._name.text(), "pth": self.pth_edit.text().strip(),
            "idx": self.idx_edit.text().strip(), "pitch": self.pitch_slider.value(),
            "index_rate": _sl_value_as_float(self.index_rate_slider),
            "rms_mix": _sl_value_as_float(self.rms_mix_slider),
            "gender": _sl_value_as_float(self.gender_slider),
            "protect": self.protect_slider.value()/100,
        }

    def set_active(self, active):
        from gui.styles import Colors
        if active:
            self._btn_use.setText("使用中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet(f"QPushButton{{background:{Colors.SUCCESS};color:white;border:none;padding:3px;border-radius:3px;font-size:11px}}")
            self.setStyleSheet(f"ModelCard{{border:1px solid {Colors.SUCCESS};border-radius:3px;margin:1px;background:{Colors.SUCCESS_BG}}}")
            self._name.setStyleSheet(f"font-weight:bold;color:{Colors.SUCCESS}")
        else:
            self._btn_use.setText("使用")
            self._btn_use.setEnabled(True)
            self._btn_use.setStyleSheet(f"QPushButton{{background:{Colors.SECONDARY};color:white;border:none;padding:3px;border-radius:3px;font-size:11px}}QPushButton:hover{{background:{Colors.SECONDARY_HOVER}}}")
            self.setStyleSheet(f"ModelCard{{border:1px solid {Colors.BORDER};border-radius:3px;margin:1px}}")
            self._name.setStyleSheet("font-weight:bold")

    def set_loading(self, loading):
        from gui.styles import Colors
        if loading:
            self._btn_use.setText("加载中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet(f"QPushButton{{background:{Colors.INFO};color:white;border:none;padding:3px;border-radius:3px;font-size:11px}}")
            self.setStyleSheet(f"ModelCard{{border:1px solid {Colors.INFO};border-radius:3px;margin:1px;background:{Colors.INFO_BG}}}")
            self._name.setStyleSheet(f"font-weight:bold;color:{Colors.INFO}")


# ─────────────────── 加载线程 ───────────────────

class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, engine, pth, idx, idx_rate):
        super().__init__()
        self.engine = engine
        self.pth = pth
        self.idx = idx
        self.rate = idx_rate
    def run(self):
        try: self.ok.emit(self.engine.load_model(self.pth, self.idx, self.rate, True))
        except Exception as e: self.err.emit(str(e))
