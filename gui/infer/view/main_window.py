"""推理 GUI 主窗口 — View 层，负责 UI 构建和信号连接。

GUI 分层后，本类承担 View 职责：
- UI 构建（_build_ui, tabs, 控制栏）
- 信号连接（按钮点击、滑动条变化）
- UI 状态更新（按钮文案、延迟显示、加载状态）
- 委托业务逻辑给 InferController

业务逻辑（参数应用、引擎控制、错误处理）由 InferController 承担。
"""
import logging
import sys

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTabWidget, QSpacerItem, QSizePolicy,
    QApplication,
)
from PySide6.QtCore import QTimer, Qt, Signal

from rvc.core.config import AppConfig
from gui.infer.controller.main_controller import InferController
from gui.infer.viewmodel.param_binding import (
    collect_gui_state,
    apply_gui_state,
    format_error_message,
    gender_to_formant,
)
from gui.infer.view.widgets import LoadThread, _sl_value_as_float
from rvc.core.config import HUBERT_DEFAULT
from gui.infer.view.tabs.audio_driver_tab import build_audio_driver_tab
from gui.infer.view.tabs.global_params_tab import build_global_params_tab
from gui.infer.view.tabs.offline_tab import build_offline_tab
from gui.infer.view.tabs.experimental_tab import build_experimental_tab
from gui.infer.controller.device_manager import DeviceManager
from gui.infer.controller.offline_manager import OfflineManager
from gui.styles import ButtonStyles, Layout

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    # 音频回调线程产生的运行时错误通过此信号转发到主线程（禁止回调线程操作 Qt）
    runtime_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC")
        self.resize(200, 150)
        self.controller = InferController(on_runtime_error=self._on_runtime_error)
        self.runtime_params = self.controller.runtime_params
        self.runtime_error.connect(self._handle_runtime_error)
        self._loading = False
        self._lt = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._build_ui()
        self._connect_runtime_param_signals()

        # 初始化管理器
        self.device_manager = DeviceManager(self)
        self.offline_manager = OfflineManager(self)

        self.device_manager.load_hostapis()
        self._load_gui_config()
        # Connect refresh button after device_manager is ready
        self.refresh_btn.clicked.connect(self._reload_dev)
        # 窗口稳定后后台预热引擎（torch 加载 ~1.6s），避免首次点「开始」卡顿
        QTimer.singleShot(300, self._warmup_engine)
        # 系统托盘：关闭=最小化到托盘；托盘不可用时直接报错退出
        self.tray = None
        try:
            from gui.infer.view.tray import TrayManager
            self.tray = TrayManager(self, on_quit=self._tray_quit)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"托盘初始化失败：{e}\n程序将退出。")
            sys.exit(1)

    def _load_gui_config(self) -> None:
        """从持久化配置加载 GUI 状态（嵌套结构 + 实验参数）。"""
        from gui.configs import load_config
        from gui.infer.viewmodel.param_binding import state_from_dict
        from rvc.core.experimental import experimental_config
        cfg = load_config()
        state = state_from_dict(cfg.get("gui", {}))
        self.apply_gui_state(state)
        # 加载实验参数
        experimental_config.from_dict(cfg.get("experimental", {}))
        # 恢复控件值时阻塞信号，避免信号处理函数用控件值覆盖 experimental_config
        self.exp_protect_transition_range.blockSignals(True)
        self.exp_rmvpe_threshold_slider.blockSignals(True)
        self.exp_fcpe_threshold_slider.blockSignals(True)
        self.exp_pitch_map_src_range.blockSignals(True)
        self.exp_pitch_map_dst_range.blockSignals(True)
        self.exp_input_neutralization_slider.blockSignals(True)
        # 过渡区域（RangeSlider 双滑块，从中心/宽度反推上下限，范围 0-50）
        _tc = experimental_config.protect_soft_threshold_hz
        _tw = experimental_config.protect_soft_width
        _t_low = max(0.0, _tc - _tw / 2)
        _t_high = min(50.0, _tc + _tw / 2)
        self.exp_protect_transition_range.setRange(_t_low, _t_high)
        if hasattr(self, 'exp_protect_transition_label'):
            self.exp_protect_transition_label.setText(f'{_t_low:.0f}-{_t_high:.0f}Hz')
        # F0 清浊阈值（DoubleSlider，直接传物理值）
        self.exp_rmvpe_threshold_slider.setValue(experimental_config.rmvpe_threshold)
        self.exp_fcpe_threshold_slider.setValue(experimental_config.fcpe_confidence_threshold)
        # 音域映射（RangeSlider 双滑块，直接传物理值）
        self.exp_pitch_map_src_range.setRange(
            experimental_config.pitch_map_src_min,
            experimental_config.pitch_map_src_max,
        )
        if hasattr(self, 'exp_pitch_map_src_label'):
            self.exp_pitch_map_src_label.setText(f'{experimental_config.pitch_map_src_min:.0f}-{experimental_config.pitch_map_src_max:.0f}Hz')
        self.exp_pitch_map_dst_range.setRange(
            experimental_config.pitch_map_dst_min,
            experimental_config.pitch_map_dst_max,
        )
        if hasattr(self, 'exp_pitch_map_dst_label'):
            self.exp_pitch_map_dst_label.setText(f'{experimental_config.pitch_map_dst_min:.0f}-{experimental_config.pitch_map_dst_max:.0f}Hz')
        # 输入语音中性化（DoubleSlider，直接传物理值）
        self.exp_input_neutralization_slider.setValue(experimental_config.input_neutralization_strength)
        # 恢复完成后解除信号阻塞
        self.exp_protect_transition_range.blockSignals(False)
        self.exp_rmvpe_threshold_slider.blockSignals(False)
        self.exp_fcpe_threshold_slider.blockSignals(False)
        self.exp_pitch_map_src_range.blockSignals(False)
        self.exp_pitch_map_dst_range.blockSignals(False)
        self.exp_input_neutralization_slider.blockSignals(False)
        # 模型路径按钮：根据 win.model_path 更新显示文件名
        if hasattr(self, "model_path") and self.model_path:
            from pathlib import Path
            self.model_path_btn.setText(Path(self.model_path).name)
            self.model_path_btn.setToolTip(self.model_path)

    def _save_gui_config(self) -> None:
        """保存当前 GUI 状态到持久化配置（嵌套结构 + 实验参数）。"""
        from gui.configs import load_config, save_config
        from gui.infer.viewmodel.param_binding import state_to_dict
        from rvc.core.experimental import experimental_config
        # 保存前从 RangeSlider 直接读取当前值，确保信号丢失时也不丢
        if hasattr(self, "exp_protect_transition_range"):
            _low = self.exp_protect_transition_range.low()
            _high = self.exp_protect_transition_range.high()
            experimental_config.protect_soft_threshold_hz = (_low + _high) / 2
            experimental_config.protect_soft_width = _high - _low
        if hasattr(self, "exp_pitch_map_src_range"):
            experimental_config.pitch_map_src_min = self.exp_pitch_map_src_range.low()
            experimental_config.pitch_map_src_max = self.exp_pitch_map_src_range.high()
        if hasattr(self, "exp_pitch_map_dst_range"):
            experimental_config.pitch_map_dst_min = self.exp_pitch_map_dst_range.low()
            experimental_config.pitch_map_dst_max = self.exp_pitch_map_dst_range.high()
        if hasattr(self, "exp_input_neutralization_slider"):
            experimental_config.input_neutralization_strength = float(self.exp_input_neutralization_slider.value())
        cfg = load_config()
        cfg["gui"] = state_to_dict(self.collect_gui_state())
        cfg["experimental"] = experimental_config.to_dict()
        save_config(cfg)

    def _tray_quit(self):
        """托盘退出：完整清理（停 timer/加载线程/离线/引擎/保存配置）后退出应用。"""
        self._timer.stop()
        if self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)
        try:
            self._save_gui_config()
        except Exception as e:
            logger.error("保存配置失败：%s", e, exc_info=True)
        self.controller.stop()
        QApplication.instance().quit()

    def _warmup_engine(self):
        """后台线程预热 engine — 首次构造会加载 torch 并做 CUDA 探测。"""
        import threading

        def _do():
            try:
                self.engine  # 触发惰性构造
                logger.info("引擎预热完成")
            except Exception:
                logger.warning("引擎预热失败（点开始时将再次尝试）", exc_info=True)

        threading.Thread(target=_do, daemon=True, name="engine-warmup").start()

    @property
    def engine(self):
        """惰性获取引擎 — 首次访问才构造（构造会加载 torch，避免拖慢窗口出现）。"""
        return self.controller.engine

    # ── UI 辅助方法 ──

    def _show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "提示", message)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "错误", message)

    def _show_info(self, message: str) -> None:
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
        tabs.addTab(build_experimental_tab(self), "高级功能")
        tabs.addTab(build_offline_tab(self), "离线推理")
        root.addWidget(tabs)

        # ── 底部控制栏 ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(Layout.SPACING_NORMAL)

        # 左侧：开始/停止（单按钮，按运行状态切换文案与颜色）
        btn_group = QHBoxLayout()
        self.btn_toggle = QPushButton("开始")
        self.btn_toggle.setFixedSize(Layout.BTN_WIDTH_NORMAL, Layout.BTN_HEIGHT_NORMAL)
        self.btn_toggle.setStyleSheet(ButtonStyles.primary())
        self.btn_toggle.clicked.connect(self._on_toggle_clicked)
        btn_group.addWidget(self.btn_toggle)

        spacer1 = QSpacerItem(40, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        btn_group.addSpacerItem(spacer1)

        # 右侧：输入音高 + 延迟显示
        self.pitch_lbl = QLabel("音高: -")
        self.pitch_lbl.setMinimumWidth(100)
        self.pitch_lbl.setToolTip("当前输入音高（Hz，实时检测）：辅助调节高级功能中的音域映射参数")
        btn_group.addWidget(self.pitch_lbl)

        self.delay_lbl = QLabel("延迟: -")
        self.delay_lbl.setMinimumWidth(120)
        self.delay_lbl.setToolTip("端到端实测延迟（含声卡缓冲）：想降延迟调小「采样长度」，或让输出设备与流采样率一致")
        btn_group.addWidget(self.delay_lbl)

        ctrl.addLayout(btn_group)
        root.addLayout(ctrl)

    def _connect_runtime_param_signals(self):
        """连接运行时参数控件的变化信号，实现引擎运行中拖动滑动条实时生效。"""
        self.protect_slider.valueChanged.connect(lambda _: self._apply_runtime_params())
        self.rms_mix_slider.valueChanged.connect(lambda _: self._apply_runtime_params())
        self.f0_rmvp_btn.toggled.connect(lambda _: self._apply_runtime_params())

    def _update_timer(self):
        if self.engine.running and self.engine.measure_ms > 0:
            self.delay_lbl.setText(f"延迟: {self.engine.measure_ms:.0f}ms")
            # 显示当前输入音高（从 f0_extractor 模块级变量获取，音域映射之前的原始值）
            from rvc.inference.f0_extractor import last_input_pitch
            if last_input_pitch > 0:
                self.pitch_lbl.setText(f"音高: {last_input_pitch:.0f}Hz")
            else:
                self.pitch_lbl.setText("音高: -")
        if self.tray is not None:
            self.tray.update_status()

    # ── 委托方法 ──

    def _reload_dev(self):
        """委托给 DeviceManager（运行中禁止刷新，防止杀活动流）"""
        if self.controller.is_running:
            self._show_warning("运行中不能刷新设备，请先停止")
            return
        self.device_manager.reload_devices()

    def _ha_changed(self, name):
        self.device_manager.on_hostapi_changed(name)

    # ── 参数应用（委托给 controller）──

    def _apply_model_params(self):
        # formant 已移到全局参数（参数调节），从全局状态读取
        state = self.collect_gui_state()
        inf = state.inference
        self.controller.apply_model_params(
            formant=inf.formant,
        )

    def _apply_runtime_params(self):
        state = self.collect_gui_state()
        inf = state.inference
        self.controller.apply_runtime_params(
            protect=inf.protect,
            f0_method=inf.f0_method,
            rms_mix=inf.rms_mix,
        )

    # ── UI 状态管理 ──

    def _set_toggle_button(self, text, enabled, style):
        self.btn_toggle.setEnabled(enabled)
        self.btn_toggle.setText(text)
        self.btn_toggle.setStyleSheet(style)

    def _reset_runtime_ui(self):
        self._timer.stop()
        self._set_toggle_button("开始", True, ButtonStyles.primary())
        self.delay_lbl.setText("延迟: -")
        self.pitch_lbl.setText("音高: -")

    def _mark_loading(self):
        self._set_toggle_button("加载中", False, ButtonStyles.secondary())

    def _mark_running(self):
        self._set_toggle_button("停止", True, ButtonStyles.danger())
        self._timer.start(200)

    def collect_gui_state(self) -> AppConfig:
        return collect_gui_state(self)

    def apply_gui_state(self, state: AppConfig) -> None:
        apply_gui_state(self, state)

    # ── 启动/停止（业务逻辑委托给 controller，UI 状态留在本类）──

    def _on_toggle_clicked(self):
        if self.engine.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        pth = self.model_path.strip() if hasattr(self, 'model_path') else ''
        if not pth:
            self._show_warning("请先在参数调节中选择模型文件")
            return
        hubert = self.hubert_combo.currentText()
        self._apply_model_params()
        self._apply_runtime_params()

        # 保存配置（在启动前保存当前设置）
        try:
            self._save_gui_config()
            logger.debug("配置已保存")
        except Exception as e:
            logger.warning("保存配置失败：%s", e)

        self._start_engine(pth, hubert)

    def _start_engine(self, pth, hubert=HUBERT_DEFAULT):
        if self._loading:
            old = self._lt
            if old and old.isRunning():
                old.request_stop()
                if not old.wait(3000):
                    logger.warning("旧模型加载线程 3s 内未结束（load_model 阻塞中），断开信号后继续")
            for sig in (old.ok, old.err, old.finished):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            old.deleteLater()
            self._loading = False

        self._loading = True
        self.controller.begin_load(None)
        self._mark_loading()
        self._lt = LoadThread(self.engine, pth, hubert)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        self.controller.end_load()
        if self._lt:
            self._lt.deleteLater()
            self._lt = None

    def _on_loaded(self, sr):
        try:
            state = self.collect_gui_state()
            eng = state.engine
            stats = self.controller.setup_engine(
                sr_mode=eng.sr_mode,
                input_device_idx=self.device_manager.get_input_device_index(self.input_combo.currentIndex()),
                output_device_idx=self.device_manager.get_output_device_index(self.output_combo.currentIndex()),
                output2_device_idx=self.device_manager.get_output_device_index(self.output2_combo.currentIndex() - 1),
                block_time=eng.block_time,
                crossfade_time=eng.crossfade_time,
                extra_time=eng.extra_time,
                enable_out2=eng.enable_out2,
            )
            self.sr_model_radio.setText(f"模型 {stats.sr_model}")
            self.sr_device_radio.setText(f"设备 {stats.sr_dev}")
            self._mark_running()
            if self.tray is not None:
                self.tray.update_status()
            self._save_gui_config()
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, e):
        self._reset_runtime_ui()
        self._show_error(format_error_message(e))

    def _on_runtime_error(self, message):
        # 在音频回调线程（PortAudio）调用：只发信号，所有 Qt 操作转发主线程执行
        self.runtime_error.emit(message)

    def _handle_runtime_error(self, message):
        # 主线程：流已在音频回调内通过 CallbackStop 安全停止；委托 controller 处理业务逻辑
        self.controller.handle_runtime_error(message)
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        self._show_error(f"实时推理错误: {message}")

    def _stop(self):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.request_stop()
                self._lt.wait(3000)
            self._loading = False
            self.controller.end_load()
            self._reset_runtime_ui()
            if self.tray is not None:
                self.tray.update_status()
            return

        if not self.engine.running:
            return
        self.controller.stop_inference()
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("停止")

    # ── 离线推理委托 ──

    def _off_browse(self, tgt, kind):
        self.offline_manager.browse_file(tgt, kind)

    def _off_start(self):
        # 保存配置（在开始转换前保存当前设置）
        try:
            self._save_gui_config()
            logger.debug("配置已保存")
        except Exception as e:
            logger.warning("保存配置失败：%s", e)
        self.offline_manager.start_conversion()

    def closeEvent(self, event):
        """点关闭按钮 → 隐藏到托盘继续运行（变声不中断），不退出。"""
        event.ignore()
        self.hide()
        if self.tray is not None:
            self.tray.notify_minimized()



