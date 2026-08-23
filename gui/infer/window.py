"""推理 GUI 主窗口"""
import logging
import sys

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTabWidget, QSpacerItem, QSizePolicy, QFileDialog,
    QApplication,
)
from PySide6.QtCore import QTimer, Qt, Signal

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
from gui.infer.tabs.offline_tab import build_offline_tab
from gui.infer.model_manager import ModelManager
from gui.infer.config_manager import ConfigManager
from gui.infer.device_manager import DeviceManager
from gui.infer.offline_manager import OfflineManager
from gui.infer.utils import format_error_message
from gui.styles import ButtonStyles, Layout

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    # 音频回调线程产生的运行时错误通过此信号转发到主线程（禁止回调线程操作 Qt）
    runtime_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self.resize(200, 150)  # 窗口大小减半
        self.controller = InferController(on_runtime_error=self._on_runtime_error)
        self.runtime_params = self.controller.runtime_params
        self.runtime_error.connect(self._handle_runtime_error)
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
        # 窗口稳定后后台预热引擎（torch 加载 ~1.6s），避免首次点「开始」卡顿
        QTimer.singleShot(300, self._warmup_engine)
        # 系统托盘：关闭=最小化到托盘；托盘不可用时直接报错退出
        self.tray = None
        try:
            from gui.infer.tray import TrayManager
            self.tray = TrayManager(self, on_quit=self._tray_quit)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"托盘初始化失败：{e}\n程序将退出。")
            sys.exit(1)

    def _tray_quit(self):
        """托盘退出：完整清理（停 timer/加载线程/离线/引擎/保存配置）后退出应用。

        不触发 engine 惰性构造（_engine 为 None 说明从未使用过，无需 stop）。
        """
        self._timer.stop()
        if self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)
        try:
            self.config_manager.save_config()
            self.model_manager.save_models()
        except Exception as e:
            logger.error("保存配置失败: %s", e, exc_info=True)
        eng = self.controller._engine
        if eng is not None:
            try:
                eng.stop()
            except Exception:
                pass
        QApplication.instance().quit()

    def _warmup_engine(self):
        """后台线程预热 engine — 首次构造会加载 torch 并做 CUDA 探测，
        挪到后台执行，等用户点「开始」时 torch 已就绪。"""
        import threading

        def _do():
            try:
                self.engine  # 触发惰性构造
                logger.info("引擎预热完成（torch 已加载）")
            except Exception:
                logger.warning("引擎预热失败（点开始时将再次尝试）", exc_info=True)

        threading.Thread(target=_do, daemon=True, name="engine-warmup").start()

    @property
    def engine(self):
        """惰性获取引擎 — 首次访问才构造（构造会加载 torch，避免拖慢窗口出现）。"""
        return self.controller.engine

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
        tabs.addTab(driver_w, "设备驱动")
        tabs.addTab(build_global_params_tab(self), "参数调节")
        tabs.addTab(build_models_tab(self), "模型列表")
        tabs.addTab(build_offline_tab(self), "离线推理")
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
        if self.engine.running and self.engine.measure_ms > 0:
            # 硬件时间戳实测（含设备缓冲），比估算更贴近真实听感
            self.delay_lbl.setText(f"延迟: {self.engine.measure_ms:.0f}ms")
        if self.tray is not None:
            self.tray.update_status()

    # ── 模型管理委托 ──

    def _add_model(self):
        """委托给 ModelManager"""
        self.model_manager.add_model_from_file()

    # ── 设备管理委托 ──

    def _reload_dev(self):
        """委托给 DeviceManager（运行中禁止刷新，防止杀活动流）"""
        eng = self.controller._engine
        if eng is not None and eng.running:
            self._show_warning("运行中不能刷新设备，请先停止")
            return
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
            old = self._lt
            if old and old.isRunning():
                old.request_stop()
                old.wait(3000)  # 超时后旧线程可能仍在跑
            # 无论旧线程是否结束，先断开其信号：防止它稍后发的 finished
            # 触发 _on_load_done 误删新线程（self._lt 已被替换）
            for sig in (old.ok, old.err, old.finished):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            old.deleteLater()
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
            if self.tray is not None:
                self.tray.update_status()
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
        # 在音频回调线程（PortAudio）调用：只发信号，所有 Qt 操作转发主线程执行
        self.runtime_error.emit(message)

    def _handle_runtime_error(self, message):
        # 主线程：流已在音频回调内通过 CallbackStop 安全停止；这里只复位 UI
        # （保留已加载模型），并延迟释放设备，避免回调线程内直接 stop 造成死锁。
        # 下次「开始」前 setup 也会兜底关闭。
        self.engine.runtime_error_pending = False
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
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
            if self.tray is not None:
                self.tray.update_status()
            return

        if not self.engine.running:
            return
        self.engine.stop()
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("停止")

    # ── 离线推理委托 ──

    def _off_browse(self, tgt, kind):
        """委托给 OfflineManager"""
        self.offline_manager.browse_file(tgt, kind)

    def _off_start(self):
        """委托给 OfflineManager"""
        self.offline_manager.start_conversion()

    def closeEvent(self, event):
        """点关闭按钮 → 隐藏到托盘继续运行（变声不中断），不退出。

        真正的退出走托盘菜单（_tray_quit），那里做完整清理。
        """
        event.ignore()
        self.hide()
        if self.tray is not None:
            self.tray.notify_minimized()