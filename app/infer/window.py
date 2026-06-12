"""推理 GUI 主窗口"""
import json
import logging
import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
    QTabWidget,
)
from PySide6.QtCore import QTimer

from rvc.params import p
from rvc.audio_utils import get_audio_devices, PRESETS
from rvc.audio_io import RealtimeEngine
from rvc.offline_worker import OfflineWorker
from app.infer.widgets import ModelCard, ModelListData, LoadThread
from app.infer.tabs.settings_tab import build_settings_tab
from app.infer.tabs.models_tab import build_models_tab
from app.infer.tabs.audio_tab import build_audio_tab
from app.infer.tabs.offline_tab import build_offline_tab

logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/inuse/gui_config.json"

engine = RealtimeEngine()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self._active_card = None
        self._loading = False
        self._off_worker = None
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: self.stat_lbl.setText(f"推理: {int(engine.infer_ms)}"))
        self._build_ui()
        self._load_cfg()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        tabs = QTabWidget()
        tabs.addTab(build_settings_tab(self), "设置")
        tabs.addTab(build_models_tab(self), "模型")
        tabs.addTab(build_audio_tab(self), "声学")
        tabs.addTab(build_offline_tab(self), "离线")
        root.addWidget(tabs)

        # 底部控制栏
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.btn_start = QPushButton("开始")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        self.model_lbl = QLabel("当前: -")
        self.model_lbl.setMinimumWidth(140)
        self.delay_lbl = QLabel("延迟: -")
        self.delay_lbl.setMinimumWidth(70)
        self.stat_lbl = QLabel("推理: -")
        self.stat_lbl.setMinimumWidth(80)
        ctrl.addWidget(self.model_lbl)
        ctrl.addStretch()
        ctrl.addWidget(self.delay_lbl)
        ctrl.addWidget(self.stat_lbl)
        root.addLayout(ctrl)

    # ── 模型管理 ──

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "assets/weights", "模型 (*.pth)")
        if not path:
            return
        name = os.path.splitext(os.path.basename(path))[0]
        self._add_card(name=name, pth=path)

    def _add_card(self, name="", pth="", idx="", pitch=0,
                  index_rate=0.0, rms_mix=0.0, gender=50, protect=50):
        card = ModelCard(name, pth, idx, pitch, index_rate=index_rate, rms_mix=rms_mix, gender=gender, protect=protect)
        card.load_requested.connect(self._on_card_load)
        card._del.clicked.connect(lambda: self._remove_card(card))
        self._models_layout.insertWidget(self._models_layout.count() - 1, card)
        self._model_cards.append(card)
        return card

    def _remove_card(self, card):
        if self._active_card == card:
            self._active_card = None
        self._model_cards.remove(card)
        self._models_layout.removeWidget(card)
        card.deleteLater()
        self._save_models()

    def _on_card_load(self, name, pth, idx, pitch, ir, rms, gender, protect):
        if not pth:
            return
        if self._active_card:
            self._active_card.set_active(False)
        for c in self._model_cards:
            if c.pth_edit.text().strip() == pth:
                self._active_card = c
                c.set_active(True)
                break
        self.model_lbl.setText(f"当前: {name}")

    def _save_models(self):
        models = [c.get_data() for c in self._model_cards]
        ModelListData.save(models)

    def _apply_preset(self, name):
        if name not in PRESETS:
            return
        pr = PRESETS[name]
        # 应用完整预设
        self.eq_sub.setValue(int(pr.get("eq_sub", 0) * 100))
        self.eq_lo.setValue(int(pr.get("eq_low", 0) * 100))
        self.eq_mi.setValue(int(pr.get("eq_mid", 0) * 100))
        self.eq_hi_mid.setValue(int(pr.get("eq_hi_mid", 0) * 100))
        self.eq_hi.setValue(int(pr.get("eq_high", 0) * 100))
        self.rev_sl.setValue(int(pr.get("reverb", 0) * 100))

    # ── 配置持久化 ──

    def _load_cfg(self):
        for m in ModelListData.load():
            self._add_card(m.get("name", ""), m.get("pth", ""), m.get("idx", ""),
                           m.get("pitch", 0),
                           m.get("index_rate", 0), m.get("rms_mix", 0),
                           int(m.get("gender", 0.5) * 100), int(m.get("protect", 0.5) * 100))
        d = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                pass
        self._reload_dev()
        ha = d.get("ha", "")
        if ha:
            idx = self.ha_combo.findText(ha)
            if idx >= 0:
                self.ha_combo.setCurrentIndex(idx)
        for dev_key, combo in [("in_dev", self.in_combo), ("out_dev", self.out_combo)]:
            dev = d.get(dev_key, "")
            if dev:
                idx = combo.findText(dev)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        out2_dev = d.get("out2_dev", "")
        if out2_dev:
            idx = self.out2_combo.findText(out2_dev)
            if idx >= 0:
                self.out2_combo.setCurrentIndex(idx)
        if d.get("sr_mode", "model") == "device":
            self.sr_r2.setChecked(True)
        f0 = d.get("f0", "fcpe")
        idx = self.f0_combo.findText(f0)
        if idx >= 0:
            self.f0_combo.setCurrentIndex(idx)
        self.th_sl.setValue(d.get("th", -60))
        self.bl_sl.setValue(int(d.get("bl", 0.25) * 100))
        self.cf_sl.setValue(int(d.get("cf", 0.05) * 100))
        self.ex_sl.setValue(int(d.get("ex", 2.5) * 100))
        self.inr.setChecked(d.get("inr", False))
        self.ounr.setChecked(d.get("ounr", False))
        self.eq_en.setChecked(d.get("eq_en", False))
        self.eq_sub.setValue(int(d.get("eq_sub", 0) * 100))
        self.eq_lo.setValue(int(d.get("eq_lo", 0) * 100))
        self.eq_mi.setValue(int(d.get("eq_mi", 0) * 100))
        self.eq_hi_mid.setValue(int(d.get("eq_hi_mid", 0) * 100))
        self.eq_hi.setValue(int(d.get("eq_hi", 0) * 100))
        self.rev_sl.setValue(int(d.get("rev", 0) * 100))
        pr = d.get("preset", "原声纯净")
        if pr in PRESETS:
            self.preset_combo.setCurrentText(pr)
        selected = d.get("selected", "")
        if selected:
            for c in self._model_cards:
                if c.pth_edit.text().strip() == selected:
                    self._active_card = c
                    c.set_active(True)
                    self.model_lbl.setText(f"当前: {c._name.text()}")
                    break

    def _save_cfg(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        self._save_models()
        d = {
            "version": 2, "th": self.th_sl.value(), "bl": self.bl_sl.value() / 100,
            "cf": self.cf_sl.value() / 100, "ex": self.ex_sl.value() / 100,
            "inr": self.inr.isChecked(), "ounr": self.ounr.isChecked(),
            "f0": self.f0_combo.currentText(),
            "eq_en": self.eq_en.isChecked(),
            "eq_sub": self.eq_sub.value() / 100, "eq_lo": self.eq_lo.value() / 100,
            "eq_mi": self.eq_mi.value() / 100, "eq_hi_mid": self.eq_hi_mid.value() / 100,
            "eq_hi": self.eq_hi.value() / 100,
            "rev": self.rev_sl.value() / 100,
            "preset": self.preset_combo.currentText(),
            "ha": self.ha_combo.currentText(),
            "in_dev": self.in_combo.currentText(),
            "out_dev": self.out_combo.currentText(),
            "out2_dev": self.out2_combo.currentText(),
            "sr_mode": "model" if self.sr_r1.isChecked() else "device",
            "selected": self._active_card.pth_edit.text().strip() if self._active_card else "",
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.warning("保存配置失败", exc_info=True)

    # ── 设备管理 ──

    def _reload_dev(self):
        self.ha_combo.blockSignals(True)
        self.ha_combo.clear()
        names, *_ = get_audio_devices()
        self.ha_combo.addItems(names)
        self.ha_combo.blockSignals(False)
        self._ha_changed(self.ha_combo.currentText())

    def _ha_changed(self, name):
        if not name:
            return
        _, ins, outs, _, _ = get_audio_devices(name)
        self.in_combo.clear()
        self.in_combo.addItems(ins)
        self.out_combo.clear()
        self.out_combo.addItems(outs)
        self.out2_combo.clear()
        self.out2_combo.addItem("不启用")
        self.out2_combo.addItems(outs)

    # ── 启动/停止 ──

    def _start(self):
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在模型列表中选择一个模型")
            return
        pth = self._active_card.pth_edit.text().strip()
        if not pth:
            QMessageBox.warning(self, "提示", "模型文件路径为空")
            return
        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value() / 100
        p.pitch = self._active_card.pit_sl.value()
        p.index_rate = ir
        p.rms_mix = self._active_card.rms_sl.value() / 100
        p.gender = (self._active_card.gen_sl.value() / 100 - 0.5) * 4
        p.protect = self._active_card.protect_sl.value() / 100
        p.f0method = self.f0_combo.currentText()
        self._start_engine(pth, idx, ir)

    def _start_engine(self, pth, idx, idx_rate):
        # 如果正在加载，取消旧的加载任务
        if self._loading:
            if hasattr(self, '_lt') and self._lt and self._lt.isRunning():
                self._lt.terminate()
                self._lt.wait()
            self._loading = False

        self._loading = True
        if self._active_card:
            self._active_card.set_loading(True)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("加载中...")
        self.btn_start.setStyleSheet("QPushButton{background:#3b82f6;color:white;border:none;padding:4px 8px;border-radius:3px}")
        self._lt = LoadThread(engine, pth, idx, idx_rate)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        if hasattr(self, '_lt') and self._lt:
            self._lt.deleteLater()
            self._lt = None

    def _on_loaded(self, sr):
        if self._active_card:
            self._active_card.set_active(True)
        try:
            _, _, _, in_idx, out_idx = get_audio_devices(self.ha_combo.currentText())
            p.threshold = self.th_sl.value()
            p.I_nr = self.inr.isChecked()
            p.O_nr = self.ounr.isChecked()
            p.use_pv = False
            p.enable_eq = self.eq_en.isChecked()
            p.eq_sub = self.eq_sub.value() / 100
            p.eq_low = self.eq_lo.value() / 100
            p.eq_mid = self.eq_mi.value() / 100
            p.eq_hi_mid = self.eq_hi_mid.value() / 100
            p.eq_high = self.eq_hi.value() / 100
            p.reverb = self.rev_sl.value() / 100
            p.bgm_enable = False
            p.enable_out2 = self.out2_combo.currentIndex() > 0
            sr_type = "sr_model" if self.sr_r1.isChecked() else "sr_device"
            engine.setup(
                sr_type,
                in_idx[self.in_combo.currentIndex()],
                out_idx[self.out_combo.currentIndex()],
                self.bl_sl.value() / 100,
                self.cf_sl.value() / 100,
                self.ex_sl.value() / 100,
                p,
            )
            engine.bgm_audio = None
            engine.bgm_ptr = 0
            if p.enable_out2:
                engine.setup_out2(out_idx[self.out2_combo.currentIndex() - 1])
            self.sr_r1_lbl.setText(f"模型采样率: {engine.sr_model}")
            self.sr_r2_lbl.setText(f"设备采样率: {engine.sr_dev}")
            dl = (engine.stream.latency[-1] if engine.stream else 0) + self.bl_sl.value() / 100 + self.cf_sl.value() / 100 + 0.01
            if p.I_nr:
                dl += min(self.cf_sl.value() / 100, 0.04)
            self.delay_lbl.setText(f"延迟: {int(dl * 1000)}")
            self.btn_start.setEnabled(False)
            self.btn_start.setText("运行中")
            self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;border:none;padding:4px 8px;border-radius:3px}")
            self.btn_stop.setEnabled(True)
            self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;border:none;padding:4px 8px;border-radius:3px}QPushButton:hover{background:#c82333}")
            self._timer.start(200)
            self._save_cfg()
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, e):
        if self._active_card:
            self._active_card.set_active(False)
        self._active_card = None
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
        QMessageBox.critical(self, "错误", str(e))

    def _stop(self):
        # 如果正在加载，取消加载线程
        if self._loading:
            if hasattr(self, '_lt') and self._lt and self._lt.isRunning():
                self._lt.terminate()
                self._lt.wait()
            self._loading = False
            self.btn_start.setEnabled(True)
            self.btn_start.setText("开始")
            self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
            self.btn_stop.setEnabled(False)
            self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
            if self._active_card:
                self._active_card.set_active(False)
            return

        if not engine.running:
            return
        self._timer.stop()
        engine.stop()
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始")
        self.btn_start.setStyleSheet("QPushButton{background:#28a745;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#218838}QPushButton:disabled{background:#555}")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton{background:#dc3545;color:white;font-weight:bold;padding:5px 20px;border-radius:3px}QPushButton:hover{background:#c82333}QPushButton:disabled{background:#555}")
        self.delay_lbl.setText("延迟: -")
        self.stat_lbl.setText("推理: -")
        logger.info("停止")

    # ── 离线推理 ──

    def _off_browse(self, tgt, kind):
        if kind == "in":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择音频", "",
                "音频 (*.wav *.mp3 *.flac *.ogg *.m4a *.wma *.aac *.opus);;所有 (*)")
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
            QMessageBox.warning(self, "提示", "请选择输入文件")
            return
        if not os.path.exists(inp):
            QMessageBox.warning(self, "提示", f"文件不存在: {inp}")
            return
        if not out:
            base, _ = os.path.splitext(inp)
            out = base + "_converted.wav"
            self.off_out.setText(out)
        if not self._active_card:
            QMessageBox.warning(self, "提示", "请先在「模型」中选择一个模型")
            return
        pth = self._active_card.pth_edit.text().strip()
        if not pth:
            QMessageBox.warning(self, "提示", "模型路径为空")
            return
        if engine.running:
            QMessageBox.warning(self, "提示", "请先停止实时变声")
            return

        idx = self._active_card.idx_edit.text().strip()
        ir = self._active_card.ir_sl.value() / 100
        pitch = self._active_card.pit_sl.value()
        f0m = self.f0_combo.currentText()
        rms = self._active_card.rms_sl.value() / 100
        protect = self._active_card.protect_sl.value() / 100

        self._off_worker = OfflineWorker(inp, out, pth, idx, ir, pitch, f0m, rms, protect)
        self._off_worker.progress.connect(self._off_progress)
        self._off_worker.finished.connect(self._off_done)
        self._off_worker.error.connect(self._off_err)
        self.off_btn.setEnabled(False)
        self.off_btn.setText("转换中...")
        self.off_progress.setValue(0)
        self._off_worker.start()

    def _off_progress(self, cur, total):
        self.off_progress.setMaximum(total)
        self.off_progress.setValue(cur)
        self.off_status.setText(f"{cur}/{total}")

    def _off_done(self, path):
        self.off_btn.setEnabled(True)
        self.off_btn.setText("开始转换")
        self.off_status.setText(f"完成: {path}")
        if self._off_worker:
            self._off_worker.wait()
            self._off_worker = None

    def _off_err(self, msg):
        self.off_btn.setEnabled(True)
        self.off_btn.setText("开始转换")
        self.off_status.setText("错误")
        if self._off_worker:
            self._off_worker.wait()
            self._off_worker = None
        QMessageBox.critical(self, "离线推理错误", msg)

    def closeEvent(self, e):
        self._timer.stop()
        if hasattr(self, '_lt') and self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)
        if self._off_worker and self._off_worker.isRunning():
            self._off_worker.quit()
            self._off_worker.wait(2000)
        try:
            self._save_cfg()
        except Exception:
            logger.warning("保存配置失败", exc_info=True)
        engine.stop()
        e.accept()
