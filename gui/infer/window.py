"""推理 GUI 主窗口"""
import logging

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTabWidget,
)
from PySide6.QtCore import QTimer

from rvc.audio import PRESETS
from gui.infer.controller import InferController, ModelConfig, RuntimeConfig, EngineConfig
from gui.infer.widgets import LoadThread, _sl_value_as_float
from gui.infer.tabs.settings_tab import build_settings_tab
from gui.infer.tabs.models_tab import build_models_tab
from gui.infer.tabs.audio_tab import build_audio_tab
from gui.infer.tabs.offline_tab import build_offline_tab
from gui.infer.model_manager import ModelManager
from gui.infer.config_manager import ConfigManager
from gui.infer.device_manager import DeviceManager
from gui.infer.offline_manager import OfflineManager
from gui.infer.utils import format_error_message
from gui.styles import ButtonStyles, Layout

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self.resize(357, 333)
        self.controller = InferController()
        self.runtime_params = self.controller.runtime_params
        self.engine = self.controller.engine
        self._loading = False
        self._lt = None
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: self.stat_lbl.setText(f"推理: {int(self.engine.infer_ms)}"))
        self._build_ui()

        # 初始化管理器
        self.model_manager = ModelManager(self, self._models_layout)
        self.model_manager.on_card_load = self._on_card_load
        self.config_manager = ConfigManager(self)
        self.device_manager = DeviceManager(self)
        self.offline_manager = OfflineManager(self)

        self.engine.signals.runtime_error.connect(self._on_runtime_error)
        self.device_manager.load_hostapis()
        self.model_manager.load_models()
        self.config_manager.load_config()

    # ── 辅助方法 ──

    def _show_warning(self, message: str) -> None:
        """显示警告对话框"""
        QMessageBox.warning(self, "提示", message)

    def _show_error(self, message: str) -> None:
        """显示错误对话框"""
        QMessageBox.critical(self, "错误", message)

    def _show_info(self, message: str) -> None:
        """显示信息对话框"""
        QMessageBox.information(self, "提示", message)

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.addTab(build_settings_tab(self), "设置")
        tabs.addTab(build_models_tab(self), "模型")
        tabs.addTab(build_audio_tab(self), "声学")
        tabs.addTab(build_offline_tab(self), "离线")
        root.addWidget(tabs)

        # 底部控制栏
        ctrl = QHBoxLayout()
        ctrl.setSpacing(Layout.SPACING_NORMAL)

        self.btn_start = QPushButton("开始")
        self.btn_start.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self.btn_start.setStyleSheet(ButtonStyles.primary())
        self.btn_start.clicked.connect(self._start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self.btn_stop.setStyleSheet(ButtonStyles.danger())
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        self.model_lbl = QLabel("当前: -")
        self.model_lbl.setMinimumWidth(100)
        self.delay_lbl = QLabel("延迟: -")
        self.delay_lbl.setMinimumWidth(60)
        self.stat_lbl = QLabel("推理: -")
        self.stat_lbl.setMinimumWidth(60)
        ctrl.addWidget(self.model_lbl)
        ctrl.addStretch()
        ctrl.addWidget(self.delay_lbl)
        ctrl.addWidget(self.stat_lbl)
        root.addLayout(ctrl)

    # ── 模型管理委托 ──

    def _add_model(self):
        """委托给 ModelManager"""
        self.model_manager.add_model_from_file()

    def _on_card_load(self, name, pth, idx, pitch, ir, rms, gender, protect):
        """模型卡片加载回调"""
        self.model_lbl.setText(f"当前: {name}")

    # ── 预设管理 ──

    def _apply_preset(self, name):
        if name not in PRESETS:
            return
        pr = PRESETS[name]
        self.eq_sub_slider.setValue(int(pr.get("eq_sub", 0) * 100))
        self.eq_low_slider.setValue(int(pr.get("eq_low", 0) * 100))
        self.eq_mid_slider.setValue(int(pr.get("eq_mid", 0) * 100))
        self.eq_hi_mid_slider.setValue(int(pr.get("eq_hi_mid", 0) * 100))
        self.eq_high_slider.setValue(int(pr.get("eq_high", 0) * 100))

    # ── 设备管理委托 ──

    def _reload_dev(self):
        """委托给 DeviceManager"""
        self.device_manager.reload_devices()

    def _ha_changed(self, name):
        """委托给 DeviceManager"""
        self.device_manager.on_hostapi_changed(name)

    # ── 引擎参数应用 ──

    def _apply_model_params(self):
        card = self.model_manager.active_card
        if not card:
            return
        self.controller.apply_model_config(
            ModelConfig(
                pitch=card.pitch_slider.value(),
                index_rate=_sl_value_as_float(card.index_rate_slider),
                rms_mix=_sl_value_as_float(card.rms_mix_slider),
                gender=(_sl_value_as_float(card.gender_slider) - 0.5) * 4,
                protect=_sl_value_as_float(card.protect_slider),
                f0method=self.f0_combo.currentText(),
            )
        )

    def _apply_runtime_params(self):
        self.controller.apply_runtime_config(
            RuntimeConfig(
                eq_en=self.eq_enable_checkbox.isChecked(),
                eq_sub=_sl_value_as_float(self.eq_sub_slider),
                eq_low=_sl_value_as_float(self.eq_low_slider),
                eq_mid=_sl_value_as_float(self.eq_mid_slider),
                eq_hi_mid=_sl_value_as_float(self.eq_hi_mid_slider),
                eq_high=_sl_value_as_float(self.eq_high_slider),
                reverb=_sl_value_as_float(self.reverb_slider),
                out2_enabled=self.output2_combo.currentIndex() > 0,
            )
        )

    # ── UI 状态管理 ──

    def _set_start_button(self, text, enabled, style):
        self.btn_start.setEnabled(enabled)
        self.btn_start.setText(text)
        if "10b981" in style or "6366f1" in style:
            if "10b981" in style:
                self.btn_start.setStyleSheet(ButtonStyles.primary())
            else:
                self.btn_start.setStyleSheet(ButtonStyles.secondary())
        else:
            self.btn_start.setStyleSheet(style)

    def _set_stop_button(self, enabled, style):
        self.btn_stop.setEnabled(enabled)
        if enabled:
            self.btn_stop.setStyleSheet(ButtonStyles.danger())
        else:
            self.btn_stop.setStyleSheet(ButtonStyles.muted())

    def _reset_runtime_ui(self):
        self._timer.stop()
        self._set_start_button("开始", True, ButtonStyles.primary())
        self._set_stop_button(False, ButtonStyles.muted())
        self.delay_lbl.setText("延迟: -")
        self.stat_lbl.setText("推理: -")

    def _mark_loading(self):
        if self.model_manager.active_card:
            self.model_manager.active_card.set_loading(True)
        self._set_start_button("加载中", False, ButtonStyles.secondary())

    def _mark_running(self):
        self._set_start_button("运行中", False, ButtonStyles.primary())
        self._set_stop_button(True, ButtonStyles.danger())

    # ── 启动/停止 ──

    def _start(self):
        if not self.model_manager.active_card:
            self._show_warning("请先在模型列表中选择一个模型")
            return
        pth = self.model_manager.active_card.pth_edit.text().strip()
        if not pth:
            self._show_warning("模型文件路径为空")
            return
        idx = self.model_manager.active_card.idx_edit.text().strip()
        ir = _sl_value_as_float(self.model_manager.active_card.index_rate_slider)
        self._apply_model_params()
        self._start_engine(pth, idx, ir)

    def _start_engine(self, pth, idx, idx_rate):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.terminate()
                self._lt.wait()
            self._loading = False

        self._loading = True
        self._mark_loading()
        self._lt = LoadThread(self.engine, pth, idx, idx_rate)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        if self._lt:
            self._lt.deleteLater()
            self._lt = None

    def _on_loaded(self, sr):
        if self.model_manager.active_card:
            self.model_manager.active_card.set_active(True)
        try:
            stats = self.controller.setup_engine(
                EngineConfig(
                    hostapi_name=self.hostapi_combo.currentText(),
                    input_device_pos=self.input_combo.currentIndex(),
                    output_device_pos=self.output_combo.currentIndex(),
                    output2_device_pos=self.output2_combo.currentIndex() - 1,
                    sr_mode="model" if self.sr_model_radio.isChecked() else "device",
                    block_time=_sl_value_as_float(self.block_time_slider),
                    crossfade_time=_sl_value_as_float(self.crossfade_slider),
                    extra_time=_sl_value_as_float(self.extra_time_slider),
                )
            )
            self.sr_model_label.setText(f"模型采样率: {stats.sr_model}")
            self.sr_device_label.setText(f"设备采样率: {stats.sr_dev}")
            self.delay_lbl.setText(f"延迟: {stats.delay_ms}")
            self._mark_running()
            self._timer.start(200)
            self.config_manager.save_config()
            self.model_manager.save_models()
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, e):
        if self.model_manager.active_card:
            self.model_manager.active_card.set_active(False)
        self.model_manager.active_card = None
        self._reset_runtime_ui()
        self._show_error(format_error_message(e))

    def _on_runtime_error(self, message):
        if self.engine.running:
            self.engine.stop()
        else:
            self.engine.runtime_error_pending = False
        self._reset_runtime_ui()
        self._show_error(f"实时推理错误: {message}")

    def _stop(self):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.terminate()
                self._lt.wait()
            self._loading = False
            self._reset_runtime_ui()
            if self.model_manager.active_card:
                self.model_manager.active_card.set_active(False)
            return

        if not self.engine.running:
            return
        self.engine.stop()
        self._reset_runtime_ui()
        logger.info("停止")

    # ── 离线推理委托 ──

    def _off_browse(self, tgt, kind):
        """委托给 OfflineManager"""
        self.offline_manager.browse_file(tgt, kind)

    def _off_start(self):
        """委托给 OfflineManager"""
        self.offline_manager.start_conversion()

    def closeEvent(self, e):
        self._timer.stop()
        if self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)
        self.offline_manager.cleanup()
        try:
            self.config_manager.save_config()
            self.model_manager.save_models()
        except OSError as e:
            logger.error("保存配置失败（文件系统错误）: %s", e)
        except Exception as e:
            logger.error("保存配置失败: %s", e, exc_info=True)
        self.engine.stop()
        e.accept()
