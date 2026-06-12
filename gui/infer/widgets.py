"""推理 GUI 通用组件 — 模型卡片、模型列表数据、加载线程"""
import os

from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal

from configs.config import load_state_json, save_state_json
from gui.styles import ButtonStyles, LabelStyles, CardStyles, Layout


def _sl(mn, mx, st, dv):
    """创建水平滑块的快捷函数"""
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv)
    return s


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

        def add_s(label, sl, lbl, row):
            bl.addWidget(QLabel(label), row, 0); bl.addWidget(sl, row, 1); bl.addWidget(lbl, row, 2)

        self.pit_sl = _sl(-16,16,1,pitch); self.pit_lbl = QLabel(str(pitch))
        self.pit_sl.valueChanged.connect(lambda v: self.pit_lbl.setText(str(v)))
        add_s("音调", self.pit_sl, self.pit_lbl, r); r+=1
        self.gen_sl = _sl(0,100,1,gender); self.gen_lbl = QLabel(f"{(gender/100-0.5)*4:+.2f}")
        self.gen_sl.valueChanged.connect(lambda v: self.gen_lbl.setText(f"{(v/100-0.5)*4:+.2f}"))
        add_s("性别", self.gen_sl, self.gen_lbl, r); r+=1
        self.ir_sl = _sl(0,100,1,int(index_rate*100)); self.ir_lbl = QLabel(f"{index_rate:.2f}")
        self.ir_sl.valueChanged.connect(lambda v: self.ir_lbl.setText(f"{v/100:.2f}"))
        add_s("索引", self.ir_sl, self.ir_lbl, r); r+=1
        self.rms_sl = _sl(0,100,1,int(rms_mix*100)); self.rms_lbl = QLabel(f"{rms_mix:.2f}")
        self.rms_sl.valueChanged.connect(lambda v: self.rms_lbl.setText(f"{v/100:.2f}"))
        add_s("响度", self.rms_sl, self.rms_lbl, r); r+=1
        self.protect_sl = _sl(0,100,1,protect); self.protect_lbl = QLabel(f"{protect/100:.2f}")
        self.protect_sl.valueChanged.connect(lambda v: self.protect_lbl.setText(f"{v/100:.2f}"))
        add_s("辅音保护", self.protect_sl, self.protect_lbl, r); r+=1

        self._del = QPushButton("删除此模型")
        self._del.setStyleSheet("QPushButton{background:#c0392b;color:white;border:none;padding:3px;border-radius:2px;font-size:11px}QPushButton:hover{background:#e74c3c}")
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
            self.pit_sl.value(), self.ir_sl.value()/100, self.rms_sl.value()/100,
            self.gen_sl.value()/100, self.protect_sl.value()/100,
        )

    def get_data(self):
        return {
            "name": self._name.text(), "pth": self.pth_edit.text().strip(),
            "idx": self.idx_edit.text().strip(), "pitch": self.pit_sl.value(),
            "index_rate": self.ir_sl.value()/100,
            "rms_mix": self.rms_sl.value()/100, "gender": self.gen_sl.value()/100,
            "protect": self.protect_sl.value()/100,
        }

    def set_active(self, active):
        if active:
            self._btn_use.setText("使用中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet("QPushButton{background:#28a745;color:white;border:none;padding:3px;border-radius:3px;font-size:11px}")
            self.setStyleSheet("ModelCard{border:1px solid #28a745;border-radius:3px;margin:1px;background:rgba(40,167,69,0.06)}")
            self._name.setStyleSheet("font-weight:bold;color:#28a745")
        else:
            self._btn_use.setText("使用")
            self._btn_use.setEnabled(True)
            self._btn_use.setStyleSheet("QPushButton{background:#3b82f6;color:white;border:none;padding:3px;border-radius:3px;font-size:11px}QPushButton:hover{background:#2563eb}")
            self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")
            self._name.setStyleSheet("font-weight:bold")

    def set_loading(self, loading):
        if loading:
            self._btn_use.setText("加载中")
            self._btn_use.setEnabled(False)
            self._btn_use.setStyleSheet("QPushButton{background:#3b82f6;color:white;border:none;padding:3px;border-radius:3px;font-size:11px}")
            self.setStyleSheet("ModelCard{border:1px solid #3b82f6;border-radius:3px;margin:1px;background:rgba(59,130,246,0.06)}")
            self._name.setStyleSheet("font-weight:bold;color:#3b82f6")


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
