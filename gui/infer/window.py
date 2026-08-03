"""推理 GUI 主窗口"""
import logging

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTabWidget, QSpacerItem, QSizePolicy, QFileDialog,
)
from PySide6.QtCore import QTimer, Qt

from rvc.audio import PRESETS
from gui.configs.infer_state import InferGuiState
from gui.infer.controller import InferController, ModelConfig, RuntimeConfig, EngineConfig
from gui.infer.param_binding import (
    collect_gui_state as bridge_collect_gui_state,
    apply_gui_state as bridge_apply_gui_state,
    runtime_from_state,
)
from gui.infer.widgets import LoadThread, _sl_value_as_float
from gui.infer.tabs.audio_driver_tab import build_audio_driver_tab
from gui.infer.tabs.global_params_tab import build_global_params_tab
from gui.infer.tabs.models_tab import build_models_tab
from gui.infer.tabs.audio_tab import build_audio_tab
from gui.infer.tabs.bgm_tab import build_bgm_tab
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
        self.resize(200, 150)  # 窗口大小减半
        self.controller = InferController(on_runtime_error=self._on_runtime_error)
        self.runtime_params = self.controller.runtime_params
        self.engine = self.controller.engine
        self._loading = False
        self._lt = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._delay_ms = 0
        self._build_ui()

        # 初始化管理器
        self.model_manager = ModelManager(self, self._models_layout)
        self.config_manager = ConfigManager(self)
        self.device_manager = DeviceManager(self)
        self.offline_manager = OfflineManager(self)

        self.device_manager.load_hostapis()
        self.model_manager.load_models()
        self.config_manager.load_config()
        # Connect refresh button after device_manager is ready
        self.refresh_btn.clicked.connect(self._reload_dev)

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

        # Create audio_driver_tab and get the refresh button
        driver_w, self.refresh_btn = build_audio_driver_tab(self)
        tabs.addTab(driver_w, "设备")
        tabs.addTab(build_global_params_tab(self), "参数")
        tabs.addTab(build_models_tab(self), "模型")
        tabs.addTab(build_audio_tab(self), "效果")
        tabs.addTab(build_bgm_tab(self), "背景")
        tabs.addTab(build_offline_tab(self), "离线")
        root.addWidget(tabs)

        # ── 底部控制栏 ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(Layout.SPACING_NORMAL)

        # 左侧：开始/停止按钮
        btn_group = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_start.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self.btn_start.setStyleSheet(ButtonStyles.primary())
        self.btn_start.clicked.connect(self._start)
        btn_group.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self.btn_stop.setStyleSheet(ButtonStyles.danger())
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_group.addWidget(self.btn_stop)

        spacer1 = QSpacerItem(40, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        btn_group.addSpacerItem(spacer1)

        # 右侧：延迟显示
        self.delay_lbl = QLabel("延迟: -")
        self.delay_lbl.setMinimumWidth(100)
        btn_group.addWidget(self.delay_lbl)

        ctrl.addLayout(btn_group)
        root.addLayout(ctrl)

    def _update_timer(self):
        self.delay_lbl.setText(f"延迟: {self._delay_ms}+{int(self.engine.infer_ms)}")

    # ── 模型管理委托 ──

    def _add_model(self):
        """委托给 ModelManager"""
        self.model_manager.add_model_from_file()

    def _apply_preset(self, name):
        if name not in PRESETS:
            return
        pr = PRESETS[name]
        self.eq_low_slider.setValue(int(pr.get("low", 0) * 100))
        self.eq_mid_slider.setValue(int(pr.get("mid", 0) * 100))
        self.eq_high_slider.setValue(int(pr.get("high", 0) * 100))

    # ── 设备管理委托 ──

    def _reload_dev(self):
        """委托给 DeviceManager"""
        self.device_manager.reload_devices()

    def _ha_changed(self, name):
        """委托给 DeviceManager"""
        self.device_manager.on_hostapi_changed(name)

    # ── 引擎参数应用 ──

    def collect_model_config(self) -> ModelConfig | None:
        card = self.model_manager.active_card
        if not card:
            return None
        return ModelConfig(
            pitch=card.pitch_slider.value(),
            index_rate=_sl_value_as_float(card.index_rate_slider),
            gender=(_sl_value_as_float(card.gender_slider) - 0.5) * 4,
            protect=_sl_value_as_float(self.protect_slider),  # 从全局参数 Tab 读取
            f0method="rmvpe" if self.f0_rmvp_btn.isChecked() else "fcpe",
        )

    def collect_runtime_config(self) -> RuntimeConfig:
        # 从完整 GUI 状态派生运行时参数（字段绑定见 param_binding.py）
        return runtime_from_state(self.collect_gui_state())

    def collect_engine_config(self) -> EngineConfig:
        return EngineConfig(
            hostapi_name=self.hostapi_combo.currentText(),
            input_device_pos=self.input_combo.currentIndex(),
            output_device_pos=self.output_combo.currentIndex(),
            output2_device_pos=self.output2_combo.currentIndex() - 1,
            sr_mode="model" if self.sr_model_radio.isChecked() else "device",
            block_time=_sl_value_as_float(self.block_time_slider),
            crossfade_time=_sl_value_as_float(self.crossfade_slider),
            extra_time=_sl_value_as_float(self.extra_time_slider),
            bgm_path=self.bgm_path_edit.text().strip(),
        )

    def _apply_model_params(self):
        config = self.collect_model_config()
        if config:
            self.controller.apply_model_config(config)

    def _apply_runtime_params(self):
        self.controller.apply_runtime_config(self.collect_runtime_config())

    # ── UI 状态管理 ──

    def _set_start_button(self, text, enabled, style):
        self.btn_start.setEnabled(enabled)
        self.btn_start.setText(text)
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

    def _mark_loading(self):
        if self.model_manager.active_card:
            self.model_manager.active_card.set_loading(True)
        self._set_start_button("加载中", False, ButtonStyles.secondary())

    def _mark_running(self):
        self._set_start_button("运行中", False, ButtonStyles.primary())
        self._set_stop_button(True, ButtonStyles.danger())
        self._timer.start(200)

    def collect_gui_state(self) -> InferGuiState:
        return bridge_collect_gui_state(self)

    def apply_gui_state(self, state: InferGuiState) -> None:
        bridge_apply_gui_state(self, state)

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
        self._apply_runtime_params()

        # 保存配置（在启动前保存当前设置）
        try:
            self.config_manager.save_config()
            self.model_manager.save_models()
            logger.info("配置已保存")
        except Exception as e:
            logger.warning("保存配置失败: %s", e)

        self._start_engine(pth, idx, ir)

    def _start_engine(self, pth, idx, idx_rate):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.request_stop()
                self._lt.wait(3000)
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
            stats = self.controller.setup_engine(self.collect_engine_config())
            self._delay_ms = stats.delay_ms
            self.sr_model_radio.setText(f"模型 {stats.sr_model}")
            self.sr_device_radio.setText(f"设备 {stats.sr_dev}")
            self._mark_running()
            self._timer.start(200)
            self.config_manager.save_config()
            self.model_manager.save_models()
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, e):
        # 保留 active_card：模型已加载（或正在加载），报错后用户直接点「开始」即可重试，
        # 无需重新选择模型。仅复位运行态 UI 并显示错误。
        self._reset_runtime_ui()
        self._show_error(format_error_message(e))

    def _on_runtime_error(self, message):
        # 流已在音频回调内通过 CallbackStop 安全停止；这里只复位 UI（保留已加载模型），
        # 并延迟到主线程释放设备，避免回调线程内直接 stop 造成死锁。下次「开始」前 setup 也会兜底关闭。
        self.engine.runtime_error_pending = False
        self._reset_runtime_ui()
        self._show_error(f"实时推理错误: {message}")
        QTimer.singleShot(0, self.engine.stop)

    def _stop(self):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.request_stop()
                self._lt.wait(3000)
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

    # ── 背景音 ──

    def _bgm_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景音频", "",
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;所有文件 (*)",
        )
        if path:
            self.bgm_path_edit.setText(path)

    # ── 离线推理委托 ──

    def _off_browse(self, tgt, kind):
        """委托给 OfflineManager"""
        self.offline_manager.browse_file(tgt, kind)

    def _off_start(self):
        """委托给 OfflineManager"""
        self.offline_manager.start_conversion()

    def closeEvent(self, event):
        self._timer.stop()
        if self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)
        self.offline_manager.cleanup()
        try:
            self.config_manager.save_config()
            self.model_manager.save_models()
        except OSError as save_err:
            logger.error("保存配置失败（文件系统错误）: %s", save_err)
        except Exception as save_err:
            logger.error("保存配置失败: %s", save_err, exc_info=True)
        self.engine.stop()
        event.accept()