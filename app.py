"""RVC 实时语音转换 — PySide6 桌面GUI"""
import json
import os
import sys
import logging

# 修复 Windows 终端中文乱码: 强制 UTF-8 输出
import io
if sys.stdout is not None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_log_handlers = [logging.StreamHandler(stream=sys.stdout)] if sys.stdout else []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QFileDialog, QMessageBox, QGroupBox, QLineEdit, QRadioButton,
    QButtonGroup, QTabWidget, QScrollArea, QFrame, QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPalette, QColor

from rvc.params import p
from rvc.audio_utils import get_audio_devices, PRESETS
from rvc.engine import RealtimeEngine
from rvc.offline_worker import OfflineWorker


CONFIG_PATH = "configs/inuse/gui_config.json"
MODELS_PATH = "configs/models.json"


# ─────────────────── 模型列表数据 ───────────────────

class ModelListData:
    """管理模型列表的持久化"""

    @staticmethod
    def load():
        if not os.path.exists(MODELS_PATH): return []
        try:
            with open(MODELS_PATH, "r", encoding="utf-8") as f: return json.load(f).get("models", [])
        except: return []

    @staticmethod
    def save(models):
        os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
        with open(MODELS_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": models}, f, indent=2, ensure_ascii=False)


# ─────────────────── 模型卡片 ───────────────────

class ModelCard(QFrame):
    """模型卡片: 点击名称选中, 箭头展开参数"""

    load_requested = Signal(str, str, str, float, float, float, float)

    def __init__(self, name="", pth="", idx="", pitch=0, f0method="fcpe",
                 index_rate=0.0, rms_mix=0.0, gender=50, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._build(name, pth, idx, pitch, f0method, index_rate, rms_mix, gender)
        self._body.setVisible(False)

    def _build(self, name, pth, idx, pitch, f0method, index_rate, rms_mix, gender):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # 头部: 单选圆圈 + 名称 + 展开箭头
        hdr = QWidget(); hdr.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(4,3,4,3)
        self._radio = QRadioButton()
        self._radio.setAutoExclusive(False)
        self._radio.clicked.connect(self._on_load)
        self._name = QLabel(name or os.path.splitext(os.path.basename(pth))[0])
        self._name.setStyleSheet("font-weight:bold")
        self._arrow = QLabel("▶"); self._arrow.setStyleSheet("color:#666;font-size:9px")
        hl.addWidget(self._radio); hl.addWidget(self._name, 1)
        hl.addWidget(self._arrow)
        root.addWidget(hdr)
        self._name.mousePressEvent = lambda e: self._on_load()
        self._arrow.mousePressEvent = lambda e: self._toggle()

        # 内容 (展开后显示)
        self._body = QWidget()
        bl = QGridLayout(self._body); bl.setContentsMargins(24,2,6,4); bl.setSpacing(2)
        r = 0
        bl.addWidget(QLabel("模型"), r, 0)
        self.pth_edit = QLineEdit(pth); bl.addWidget(self.pth_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(22,20)
        b.clicked.connect(lambda: self._browse(self.pth_edit, "模型 (*.pth)")); bl.addWidget(b, r, 2); r+=1
        bl.addWidget(QLabel("索引"), r, 0)
        self.idx_edit = QLineEdit(idx); bl.addWidget(self.idx_edit, r, 1)
        b = QPushButton("…"); b.setFixedSize(22,20)
        b.clicked.connect(lambda: self._browse(self.idx_edit, "索引 (*.index)")); bl.addWidget(b, r, 2); r+=1

        def add_s(label, sl, lbl, row):
            bl.addWidget(QLabel(label), row, 0); bl.addWidget(sl, row, 1); bl.addWidget(lbl, row, 2)

        self.pit_sl = self._sl(-16,16,1,pitch); self.pit_lbl = QLabel(str(pitch))
        self.pit_sl.valueChanged.connect(lambda v: self.pit_lbl.setText(str(v)))
        add_s("音调", self.pit_sl, self.pit_lbl, r); r+=1
        self.gen_sl = self._sl(0,100,1,gender); self.gen_lbl = QLabel(f"{(gender/100-0.5)*4:+.2f}")
        self.gen_sl.valueChanged.connect(lambda v: self.gen_lbl.setText(f"{(v/100-0.5)*4:+.2f}"))
        add_s("性别", self.gen_sl, self.gen_lbl, r); r+=1
        self.ir_sl = self._sl(0,100,1,int(index_rate*100)); self.ir_lbl = QLabel(f"{index_rate:.2f}")
        self.ir_sl.valueChanged.connect(lambda v: self.ir_lbl.setText(f"{v/100:.2f}"))
        add_s("索引", self.ir_sl, self.ir_lbl, r); r+=1
        self.rms_sl = self._sl(0,100,1,int(rms_mix*100)); self.rms_lbl = QLabel(f"{rms_mix:.2f}")
        self.rms_sl.valueChanged.connect(lambda v: self.rms_lbl.setText(f"{v/100:.2f}"))
        add_s("响度", self.rms_sl, self.rms_lbl, r); r+=1

        self._del = QPushButton("删除此模型")
        self._del.setStyleSheet("QPushButton{background:#c0392b;color:white;border:none;padding:3px;border-radius:2px;font-size:11px}QPushButton:hover{background:#e74c3c}")
        bl.addWidget(self._del, r, 0, 1, 3)
        root.addWidget(self._body)
        self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")

    def _sl(self, mn, mx, st, dv):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv); return s

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")

    def _browse(self, tgt, filt):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filt)
        if path: tgt.setText(path)

    def _on_load(self):
        self._radio.setChecked(True)
        self.load_requested.emit(
            self._name.text(), self.pth_edit.text().strip(), self.idx_edit.text().strip(),
            self.pit_sl.value(), self.ir_sl.value()/100, self.rms_sl.value()/100,
            self.gen_sl.value()/100,
        )

    def get_data(self):
        return {
            "name": self._name.text(), "pth": self.pth_edit.text().strip(),
            "idx": self.idx_edit.text().strip(), "pitch": self.pit_sl.value(),
            "index_rate": self.ir_sl.value()/100,
            "rms_mix": self.rms_sl.value()/100, "gender": self.gen_sl.value()/100,
        }

    def set_active(self, active):
        self._radio.blockSignals(True); self._radio.setChecked(active); self._radio.blockSignals(False)
        if active:
            self.setStyleSheet("ModelCard{border:1px solid #28a745;border-radius:3px;margin:1px;background:rgba(40,167,69,0.06)}")
            self._name.setStyleSheet("font-weight:bold;color:#28a745")
        else:
            self.setStyleSheet("ModelCard{border:1px solid #444;border-radius:3px;margin:1px}")
            self._name.setStyleSheet("font-weight:bold")



# ─────────────────── 实时引擎 ───────────────────

engine = RealtimeEngine()


class LoadThread(QThread):
    ok = Signal(int); err = Signal(str)
    def __init__(self, pth, idx, idx_rate): super().__init__(); self.pth=pth; self.idx=idx; self.rate=idx_rate
    def run(self):
        try: self.ok.emit(engine.load_model(self.pth, self.idx, self.rate, True))
        except Exception as e: self.err.emit(str(e))



# ─────────────────── 主窗口 ───────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self._active_card = None
        self._loading = False
        self._off_worker = None
        self._timer = QTimer(); self._timer.timeout.connect(lambda: self.stat_lbl.setText(f"推理: {int(engine.infer_ms)}"))
        self._build_ui()
        self._load_cfg()

    def _sl(self, mn, mx, st, dv):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mn, mx); s.setSingleStep(st); s.setValue(dv); return s

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(4); root.setContentsMargins(6,6,6,6)

        tabs = QTabWidget()
        tabs.addTab(self._build_settings_tab(), "设置")
        tabs.addTab(self._build_models_tab(), "模型")
        tabs.addTab(self._build_audio_tab(), "声学")
        tabs.addTab(self._build_offline_tab(), "离线")
        root.addWidget(tabs)

        # 底部控制栏
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self.btn_start = QPushButton("开始")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_start); ctrl.addWidget(self.btn_stop)
        self.model_lbl = QLabel("当前: -"); self.model_lbl.setMinimumWidth(140)
        self.delay_lbl = QLabel("延迟: -"); self.delay_lbl.setMinimumWidth(70)
        self.stat_lbl = QLabel("推理: -"); self.stat_lbl.setMinimumWidth(80)
        ctrl.addWidget(self.model_lbl); ctrl.addStretch()
        ctrl.addWidget(self.delay_lbl); ctrl.addWidget(self.stat_lbl)
        root.addLayout(ctrl)

    def _build_settings_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1)
        r = 0

        # 设备
        g.addWidget(QLabel("音频驱动"), r, 0)
        self.ha_combo = QComboBox(); self.ha_combo.currentTextChanged.connect(self._ha_changed)
        g.addWidget(self.ha_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("麦克风"), r, 0)
        self.in_combo = QComboBox(); g.addWidget(self.in_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("主输出"), r, 0)
        self.out_combo = QComboBox(); g.addWidget(self.out_combo, r, 1, 1, 2); r+=1
        g.addWidget(QLabel("副输出"), r, 0)
        self.out2_combo = QComboBox(); g.addWidget(self.out2_combo, r, 1)
        br = QPushButton("刷新"); br.setFixedWidth(45); br.clicked.connect(self._reload_dev)
        g.addWidget(br, r, 2); r+=1
        self.sr_r1 = QRadioButton(); self.sr_r1.setChecked(True)
        self.sr_r2 = QRadioButton()
        self.sr_r1_lbl = QLabel("模型采样率: -")
        self.sr_r2_lbl = QLabel("设备采样率: -")
        sr = QHBoxLayout(); sr.setSpacing(4)
        sr.addWidget(self.sr_r1); sr.addWidget(self.sr_r1_lbl)
        sr.addSpacing(12); sr.addWidget(self.sr_r2); sr.addWidget(self.sr_r2_lbl); sr.addStretch()
        g.addLayout(sr, r, 0, 1, 3); r+=1

        # 分隔
        g.addWidget(self._sep(), r, 0, 1, 3); r+=1

        # 引擎参数
        def add_sl(label, sl, lbl, row):
            g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

        self.th_sl = self._sl(-60,0,1,-60); self.th_lbl = QLabel("-60"); self.th_lbl.setMinimumWidth(35)
        self.th_sl.valueChanged.connect(lambda v: (self.th_lbl.setText(str(v)), setattr(p,'threshold',v)))
        add_sl("响应阈值", self.th_sl, self.th_lbl, r); r+=1
        self.bl_sl = self._sl(2,150,1,25); self.bl_lbl = QLabel("0.25"); self.bl_lbl.setMinimumWidth(35)
        self.bl_sl.valueChanged.connect(lambda v: self.bl_lbl.setText(f"{v/100:.2f}"))
        add_sl("采样长度", self.bl_sl, self.bl_lbl, r); r+=1
        self.cf_sl = self._sl(1,15,1,5); self.cf_lbl = QLabel("0.05"); self.cf_lbl.setMinimumWidth(35)
        self.cf_sl.valueChanged.connect(lambda v: self.cf_lbl.setText(f"{v/100:.2f}"))
        add_sl("淡入淡出", self.cf_sl, self.cf_lbl, r); r+=1
        self.ex_sl = self._sl(5,500,1,250); self.ex_lbl = QLabel("2.50"); self.ex_lbl.setMinimumWidth(35)
        self.ex_sl.valueChanged.connect(lambda v: self.ex_lbl.setText(f"{v/100:.2f}"))
        add_sl("额外上下文", self.ex_sl, self.ex_lbl, r); r+=1

        g.addWidget(QLabel("音高算法"), r, 0)
        self.f0_combo = QComboBox(); self.f0_combo.addItems(["fcpe", "rmvpe"]); self.f0_combo.setFixedWidth(80)
        g.addWidget(self.f0_combo, r, 1)
        return w

    def _build_models_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(4); l.setContentsMargins(10,10,10,10)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("模型列表")); bar.addStretch()
        btn_add = QPushButton("+ 添加模型")
        btn_add.setStyleSheet("QPushButton{background:#007acc;color:white;padding:3px 10px;border-radius:2px;font-weight:bold}QPushButton:hover{background:#005f9e}")
        btn_add.clicked.connect(self._add_model)
        bar.addWidget(btn_add)
        l.addLayout(bar)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        self._models_layout = QVBoxLayout(container); self._models_layout.addStretch()
        scroll.setWidget(container)
        l.addWidget(scroll, 1)
        self._model_cards = []
        return w

    def _build_audio_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1)
        r = 0

        # 降噪
        self.inr = QCheckBox("输入降噪"); self.ounr = QCheckBox("输出降噪")
        nr = QHBoxLayout(); nr.addWidget(self.inr); nr.addWidget(self.ounr); nr.addStretch()
        g.addLayout(nr, r, 0, 1, 3); r+=1
        g.addWidget(self._sep(), r, 0, 1, 3); r+=1

        # 音效
        self.eq_en = QCheckBox("开启音效")
        self.preset_combo = QComboBox(); self.preset_combo.addItems(PRESETS.keys()); self.preset_combo.setFixedWidth(90)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        row0 = QHBoxLayout(); row0.addWidget(self.eq_en); row0.addWidget(self.preset_combo); row0.addStretch()
        g.addLayout(row0, r, 0, 1, 3); r+=1

        def add_eq(label, sl, lbl, row):
            g.addWidget(QLabel(label), row, 0); g.addWidget(sl, row, 1); g.addWidget(lbl, row, 2)

        self.eq_lo = self._sl(-3000,2000,500,0); self.eq_lo_lbl = QLabel("0.0"); self.eq_lo_lbl.setMinimumWidth(35)
        self.eq_lo.valueChanged.connect(lambda v: self.eq_lo_lbl.setText(f"{v/100:.1f}"))
        add_eq("低频增益", self.eq_lo, self.eq_lo_lbl, r); r+=1
        self.eq_mi = self._sl(-2000,2000,500,0); self.eq_mi_lbl = QLabel("0.0"); self.eq_mi_lbl.setMinimumWidth(35)
        self.eq_mi.valueChanged.connect(lambda v: self.eq_mi_lbl.setText(f"{v/100:.1f}"))
        add_eq("中频增益", self.eq_mi, self.eq_mi_lbl, r); r+=1
        self.eq_hi = self._sl(-3000,3000,500,0); self.eq_hi_lbl = QLabel("0.0"); self.eq_hi_lbl.setMinimumWidth(35)
        self.eq_hi.valueChanged.connect(lambda v: self.eq_hi_lbl.setText(f"{v/100:.1f}"))
        add_eq("高频增益", self.eq_hi, self.eq_hi_lbl, r); r+=1
        self.warm_sl = self._sl(0,100,1,0); self.warm_lbl = QLabel("0.00"); self.warm_lbl.setMinimumWidth(35)
        self.warm_sl.valueChanged.connect(lambda v: self.warm_lbl.setText(f"{v/100:.2f}"))
        add_eq("电子管饱和", self.warm_sl, self.warm_lbl, r); r+=1
        self.comp_sl = self._sl(0,100,5,0); self.comp_lbl = QLabel("0.00"); self.comp_lbl.setMinimumWidth(35)
        self.comp_sl.valueChanged.connect(lambda v: self.comp_lbl.setText(f"{v/100:.2f}"))
        add_eq("动态压限", self.comp_sl, self.comp_lbl, r); r+=1
        self.rev_sl = self._sl(0,50,1,0); self.rev_lbl = QLabel("0.00"); self.rev_lbl.setMinimumWidth(35)
        self.rev_sl.valueChanged.connect(lambda v: self.rev_lbl.setText(f"{v/100:.2f}"))
        add_eq("空间混响", self.rev_sl, self.rev_lbl, r)
        return w

    def _build_offline_tab(self):
        w = QWidget(); g = QGridLayout(w); g.setSpacing(6); g.setContentsMargins(10,10,10,10)
        g.setColumnStretch(1, 1); r = 0

        g.addWidget(QLabel("输入文件"), r, 0)
        self.off_in = QLineEdit(); g.addWidget(self.off_in, r, 1)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: self._off_browse(self.off_in, "in"))
        g.addWidget(b, r, 2); r += 1

        g.addWidget(QLabel("输出文件"), r, 0)
        self.off_out = QLineEdit(); g.addWidget(self.off_out, r, 1)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: self._off_browse(self.off_out, "out"))
        g.addWidget(b, r, 2); r += 1

        row = QHBoxLayout()
        self.off_btn = QPushButton("开始转换")
        self.off_btn.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:5px 16px;border-radius:3px}QPushButton:hover{background:#005f9e}QPushButton:disabled{background:#555}")
        self.off_btn.clicked.connect(self._off_start)
        row.addWidget(self.off_btn)
        self.off_status = QLabel("")
        row.addWidget(self.off_status)
        row.addStretch()
        g.addLayout(row, r, 0, 1, 3); r += 1

        self.off_progress = QProgressBar(); self.off_progress.setValue(0)
        g.addWidget(self.off_progress, r, 0, 1, 3)
        return w

    @staticmethod
    def _sep():
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet("color:#444"); return f

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "assets/weights", "模型 (*.pth)")
        if not path: return
        name = os.path.splitext(os.path.basename(path))[0]
        self._add_card(name=name, pth=path)

    def _add_card(self, name="", pth="", idx="", pitch=0, f0method="fcpe",
                  index_rate=0.0, rms_mix=0.0, gender=50):
        card = ModelCard(name, pth, idx, pitch, f0method, index_rate, rms_mix, gender)
        card.load_requested.connect(self._on_card_load)
        card._del.clicked.connect(lambda: self._remove_card(card))
        self._models_layout.insertWidget(self._models_layout.count()-1, card)
        self._model_cards.append(card)
        return card

    def _remove_card(self, card):
        if self._active_card == card: self._active_card = None
        self._model_cards.remove(card)
        self._models_layout.removeWidget(card)
        card.deleteLater()
        self._save_models()

    def _on_card_load(self, name, pth, idx, pitch, ir, rms, gender):
        if not pth: return
        # 只选中, 不加载. 点击"开始"时才加载
        if self._active_card: self._active_card.set_active(False)
        for c in self._model_cards:
            if c.pth_edit.text().strip() == pth:
                self._active_card = c; c.set_active(True); break
        self.model_lbl.setText(f"当前: {name}")

    def _save_models(self):
        models = [c.get_data() for c in self._model_cards]
        ModelListData.save(models)

    def _apply_preset(self, name):
        if name not in PRESETS: return
        pr = PRESETS[name]
        self.eq_lo.setValue(int(pr["eq_low"]*100)); self.eq_mi.setValue(int(pr["eq_mid"]*100)); self.eq_hi.setValue(int(pr["eq_high"]*100))
        self.warm_sl.setValue(int(pr["warmth"]*100)); self.comp_sl.setValue(int(pr["compress"]*100)); self.rev_sl.setValue(int(pr["reverb"]*100))

    # ── 配置持久化 ──

    def _load_cfg(self):
        # 加载模型列表
        for m in ModelListData.load():
            self._add_card(m.get("name",""), m.get("pth",""), m.get("idx",""),
                           m.get("pitch",0), m.get("f0method","fcpe"),
                           m.get("index_rate",0), m.get("rms_mix",0),
                           int(m.get("gender",0.5)*100))
        # 加载全局配置
        d = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f: d = json.load(f)
            except: pass
        # 恢复设备路由（需要先填充设备列表）
        self._reload_dev()
        ha = d.get("ha", "")
        if ha:
            idx = self.ha_combo.findText(ha)
            if idx >= 0: self.ha_combo.setCurrentIndex(idx)
        for dev_key, combo in [("in_dev", self.in_combo), ("out_dev", self.out_combo)]:
            dev = d.get(dev_key, "")
            if dev:
                idx = combo.findText(dev)
                if idx >= 0: combo.setCurrentIndex(idx)
        # 副输出 (索引0="不启用", 所以+1偏移)
        out2_dev = d.get("out2_dev", "")
        if out2_dev:
            idx = self.out2_combo.findText(out2_dev)
            if idx >= 0: self.out2_combo.setCurrentIndex(idx)
        if d.get("sr_mode", "model") == "device": self.sr_r2.setChecked(True)
        # F0 算法
        f0 = d.get("f0", "fcpe")
        idx = self.f0_combo.findText(f0)
        if idx >= 0: self.f0_combo.setCurrentIndex(idx)
        self.th_sl.setValue(d.get("th", -60))
        self.bl_sl.setValue(int(d.get("bl", 0.25)*100))
        self.cf_sl.setValue(int(d.get("cf", 0.05)*100))
        self.ex_sl.setValue(int(d.get("ex", 2.5)*100))
        self.inr.setChecked(d.get("inr", False)); self.ounr.setChecked(d.get("ounr", False))
        self.eq_en.setChecked(d.get("eq_en", False))
        self.eq_lo.setValue(int(d.get("eq_lo", 0)*100)); self.eq_mi.setValue(int(d.get("eq_mi", 0)*100))
        self.eq_hi.setValue(int(d.get("eq_hi", 0)*100)); self.warm_sl.setValue(int(d.get("warm", 0)*100))
        self.comp_sl.setValue(int(d.get("comp", 0)*100)); self.rev_sl.setValue(int(d.get("rev", 0)*100))
        pr = d.get("preset", "原声纯净")
        if pr in PRESETS: self.preset_combo.setCurrentText(pr)
        # 恢复选中的模型
        selected = d.get("selected", "")
        if selected:
            for c in self._model_cards:
                if c.pth_edit.text().strip() == selected:
                    self._active_card = c; c.set_active(True)
                    self.model_lbl.setText(f"当前: {c._name.text()}")
                    break

    def _save_cfg(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        self._save_models()
        d = {
            "version": 2, "th": self.th_sl.value(), "bl": self.bl_sl.value()/100,
            "cf": self.cf_sl.value()/100, "ex": self.ex_sl.value()/100,
            "inr": self.inr.isChecked(), "ounr": self.ounr.isChecked(),
            "f0": self.f0_combo.currentText(),
            "eq_en": self.eq_en.isChecked(),
            "eq_lo": self.eq_lo.value()/100, "eq_mi": self.eq_mi.value()/100,
            "eq_hi": self.eq_hi.value()/100, "warm": self.warm_sl.value()/100,
            "comp": self.comp_sl.value()/100, "rev": self.rev_sl.value()/100,
            "preset": self.preset_combo.currentText(),
            "ha": self.ha_combo.currentText(),
            "in_dev": self.in_combo.currentText(),
            "out_dev": self.out_combo.currentText(),
            "out2_dev": self.out2_combo.currentText(),
            "sr_mode": "model" if self.sr_r1.isChecked() else "device",
            "selected": self._active_card.pth_edit.text().strip() if self._active_card else "",
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, ensure_ascii=False)

    # ── 设备管理 ──

    def _reload_dev(self):
        self.ha_combo.blockSignals(True); self.ha_combo.clear()
        names, *_ = get_audio_devices(); self.ha_combo.addItems(names)
        self.ha_combo.blockSignals(False); self._ha_changed(self.ha_combo.currentText())

    def _ha_changed(self, name):
        if not name: return
        _, ins, outs, _, _ = get_audio_devices(name)
        self.in_combo.clear(); self.in_combo.addItems(ins)
        self.out_combo.clear(); self.out_combo.addItems(outs)
        self.out2_combo.clear()
        self.out2_combo.addItem("不启用")
        self.out2_combo.addItems(outs)

    # ── 启动/停止 ──

    def _start(self):
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在模型列表中选择一个模型")
            return
        pth = self._active_card.pth_edit.text().strip()
        if not pth: QMessageBox.warning(self, "提示", "模型文件路径为空"); return
        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value()/100
        p.pitch = self._active_card.pit_sl.value()
        p.index_rate = ir; p.rms_mix = self._active_card.rms_sl.value()/100
        p.gender = (self._active_card.gen_sl.value()/100 - 0.5) * 4  # 0~100 → -2~+2 半音
        p.f0method = self.f0_combo.currentText()
        name = self._active_card._name.text()
        # logger.info("选择: %s (pitch=%d, f0=%s, ir=%.2f)", name, p.pitch, p.f0method, ir)
        self._start_engine(pth, idx, ir)

    def _start_engine(self, pth, idx, idx_rate):
        if self._loading: return
        self._loading = True
        self.btn_start.setEnabled(False); self.btn_start.setText("加载中...")
        self._lt = LoadThread(pth, idx, idx_rate)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        if hasattr(self, '_lt') and self._lt:
            self._lt.deleteLater(); self._lt = None

    def _on_loaded(self, sr):
        try:
            _, _, _, in_idx, out_idx = get_audio_devices(self.ha_combo.currentText())
            p.threshold = self.th_sl.value()
            p.I_nr = self.inr.isChecked(); p.O_nr = self.ounr.isChecked(); p.use_pv = False
            p.enable_eq = self.eq_en.isChecked(); p.eq_low = self.eq_lo.value()/100
            p.eq_mid = self.eq_mi.value()/100; p.eq_hi = self.eq_hi.value()/100
            p.warmth = self.warm_sl.value()/100; p.compress = self.comp_sl.value()/100
            p.reverb = self.rev_sl.value()/100
            p.bgm_enable = False; p.enable_out2 = self.out2_combo.currentIndex() > 0
            sr_type = "sr_model" if self.sr_r1.isChecked() else "sr_device"
            engine.setup(sr_type, in_idx[self.in_combo.currentIndex()], out_idx[self.out_combo.currentIndex()],
                         self.bl_sl.value()/100, self.cf_sl.value()/100, self.ex_sl.value()/100, p)
            engine.bgm_audio = None; engine.bgm_ptr = 0
            if p.enable_out2:
                engine.setup_out2(out_idx[self.out2_combo.currentIndex() - 1])
            self.sr_r1_lbl.setText(f"模型采样率: {engine.sr_model}")
            self.sr_r2_lbl.setText(f"设备采样率: {engine.sr_dev}")
            dl = (engine.stream.latency[-1] if engine.stream else 0) + self.bl_sl.value()/100 + self.cf_sl.value()/100 + 0.01
            if p.I_nr: dl += min(self.cf_sl.value()/100, 0.04)
            self.delay_lbl.setText(f"延迟: {int(dl*1000)}")
            self.btn_start.setEnabled(False); self.btn_start.setText("运行中")
            self.btn_stop.setEnabled(True)
            self._timer.start(200); self._save_cfg()
            # logger.info("启动完成 (设备:%d 模型:%d)", engine.sr_dev, engine.sr_model)
        except Exception as e: self._on_err(str(e))

    def _on_err(self, e):
        self.btn_start.setEnabled(True); self.btn_start.setText("开始")
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "错误", str(e))

    def _stop(self):
        if not engine.running: return
        self._timer.stop(); engine.stop()
        self.btn_start.setEnabled(True); self.btn_start.setText("开始")
        self.btn_stop.setEnabled(False)
        self.delay_lbl.setText("延迟: -"); self.stat_lbl.setText("推理: -")
        logger.info("停止")

    # ── 离线推理 ──

    def _off_browse(self, tgt, kind):
        if kind == "in":
            path, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频 (*.wav *.mp3 *.flac *.ogg *.m4a *.wma *.aac *.opus);;所有 (*)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存音频", "", "WAV (*.wav)")
        if path:
            tgt.setText(path)
            if kind == "in" and not self.off_out.text():
                base, _ = os.path.splitext(path)
                self.off_out.setText(base + "_converted.wav")

    def _off_start(self):
        inp = self.off_in.text().strip()
        out = self.off_out.text().strip()
        if not inp:
            QMessageBox.warning(self, "提示", "请选择输入文件"); return
        if not os.path.exists(inp):
            QMessageBox.warning(self, "提示", f"文件不存在: {inp}"); return
        if not out:
            base, _ = os.path.splitext(inp); out = base + "_converted.wav"
            self.off_out.setText(out)
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在「模型」中选择一个模型"); return
        pth = self._active_card.pth_edit.text().strip()
        if not pth:
            QMessageBox.warning(self, "提示", "模型路径为空"); return
        if engine.running:
            QMessageBox.warning(self, "提示", "请先停止实时变声"); return

        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value() / 100
        pitch = self._active_card.pit_sl.value()
        f0m = self.f0_combo.currentText()
        rms = self._active_card.rms_sl.value() / 100

        self._off_worker = OfflineWorker(inp, out, pth, idx, ir, pitch, f0m, rms)
        self._off_worker.progress.connect(self._off_progress)
        self._off_worker.finished.connect(self._off_done)
        self._off_worker.error.connect(self._off_err)
        self.off_btn.setEnabled(False); self.off_btn.setText("转换中...")
        self.off_progress.setValue(0)
        self._off_worker.start()

    def _off_progress(self, cur, total):
        self.off_progress.setMaximum(total)
        self.off_progress.setValue(cur)
        self.off_status.setText(f"{cur}/{total}")

    def _off_done(self, path):
        self.off_btn.setEnabled(True); self.off_btn.setText("开始转换")
        self.off_status.setText(f"完成: {path}")
        if self._off_worker:
            self._off_worker.wait(); self._off_worker = None

    def _off_err(self, msg):
        self.off_btn.setEnabled(True); self.off_btn.setText("开始转换")
        self.off_status.setText("错误")
        if self._off_worker:
            self._off_worker.wait(); self._off_worker = None
        QMessageBox.critical(self, "离线推理错误", msg)

    def closeEvent(self, e):
        if self._off_worker and self._off_worker.isRunning():
            self._off_worker.quit(); self._off_worker.wait()
        self._save_cfg(); engine.stop(); e.accept()


def set_dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(40,40,40)); p.setColor(QPalette.WindowText, QColor(220,220,220))
    p.setColor(QPalette.Base, QColor(30,30,30)); p.setColor(QPalette.Text, QColor(220,220,220))
    p.setColor(QPalette.Button, QColor(55,55,55)); p.setColor(QPalette.ButtonText, QColor(220,220,220))
    p.setColor(QPalette.Highlight, QColor(66,133,244)); p.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(p)
    app.setStyleSheet("QGroupBox{font-weight:bold;margin-top:6px}QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px}QSlider{min-height:18px}QSlider::groove:horizontal{height:4px}QSlider::handle:horizontal{width:12px;margin:-5px 0}")

if __name__ == "__main__":
    app = QApplication(sys.argv); set_dark(app)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
