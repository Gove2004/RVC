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



from rvc.core.config import InferenceParams

from gui.infer.controller.engine_controller import InferController

from gui.infer.state.bindings import (
    collect_params,
    apply_params,
)

from gui.infer.view.widgets import LoadThread, _sl_value_as_float

from rvc.core.config import HUBERT_DEFAULT

from gui.infer.view.tabs.audio_driver_tab import build_audio_driver_tab

from gui.infer.view.tabs.global_params_tab import build_global_params_tab

from gui.infer.view.tabs.offline_tab import build_offline_tab

from gui.infer.view.tabs.experimental_tab import build_experimental_tab

from gui.infer.controller.device_controller import DeviceManager

from gui.infer.controller.offline_controller import OfflineManager

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

        self._connect_all_param_signals()



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

        """从持久化配置加载 GUI 状态（嵌套结构 + 实验参数，向后兼容旧格式）。"""

        from gui.configs import load_config

        from gui.infer.state.bindings import params_from_dict

        cfg = load_config()

        # 向后兼容：旧格式有独立的 "experimental" 部分，合并到 "gui" 部分

        gui_data = cfg.get("gui", {})

        if "experimental" in cfg and "inference" not in gui_data:

            exp = cfg["experimental"]

            gui_data = dict(gui_data)

            gui_data["inference"] = {

                "rmvpe_threshold": exp.get("rmvpe_threshold", 0.05),

                "fcpe_confidence_threshold": exp.get("fcpe_confidence_threshold", 0.05),



                "pitch_map_src_min": exp.get("pitch_map_src_min", 100.0),

                "pitch_map_src_max": exp.get("pitch_map_src_max", 500.0),

                "pitch_map_dst_min": exp.get("pitch_map_dst_min", 200.0),

                "pitch_map_dst_max": exp.get("pitch_map_dst_max", 800.0),

            }

        state = params_from_dict(gui_data)
        # 原地更新 runtime_params 字段（不替换对象引用，engine/runner 自动生效）
        self.runtime_params.update_from(state)
        # apply_params 把控件值设置为 params 的值
        self.apply_gui_state(state)

        # 模型路径按钮：根据 win.model_path 更新显示文件名

        if hasattr(self, "model_path") and self.model_path:

            from pathlib import Path

            self.model_path_btn.setText(Path(self.model_path).name)

            self.model_path_btn.setToolTip(self.model_path)



    def _save_gui_config(self) -> None:

        """保存当前 GUI 状态到持久化配置（嵌套结构，含实验参数）。"""

        from gui.configs import load_config, save_config

        from gui.infer.state.bindings import params_to_dict

        cfg = load_config()

        # collect_gui_state 会从控件收集实验参数

        cfg["gui"] = params_to_dict(self.collect_gui_state())

        # 删除旧格式的 "experimental" 部分（已合并到 "gui.inference"）

        cfg.pop("experimental", None)

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
            except Exception as e:
                logger.warning("引擎预热失败（点开始时将再次尝试）", exc_info=True)
                # 无 GPU 等致命错误：主线程弹友好提示（后台线程不能直接操作 Qt）
                msg = str(e)
                QTimer.singleShot(0, lambda: self._show_error(f"引擎初始化失败: {msg}\n\n请确认已安装 NVIDIA 显卡驱动并支持 CUDA。"))

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



    def _connect_all_param_signals(self) -> None:
        """统一连接所有参数控件的变化信号 → 实时更新 runtime_params。

        覆盖 BINDINGS 表中的所有控件类型（FLOAT/INT/COMBO/CHECK/RADIO_F0/RADIO_SR），
        以及特殊控件（RangeSlider、exp_f0_threshold_slider）。
        使用 _signals_connected 标志确保只连接一次，避免重复连接。
        """
        if getattr(self, '_signals_connected', False):
            return
        self._signals_connected = True

        from gui.infer.state.bindings import (
            BINDINGS, FLOAT, INT, COMBO, CHECK, RADIO_F0, RADIO_SR, _set_nested,
        )

        # 1. BINDINGS 表中的所有控件
        for path, widget, kind, _default in BINDINGS:
            if not widget or not hasattr(self, widget):
                continue
            w = getattr(self, widget)
            if kind in (FLOAT, INT):
                # 注意：DoubleSlider 的 valueChanged 发射内部编码整数值，
                # 所以必须读取 w.value() 获取物理值，而不是用信号参数
                def _on_val(_v, p=path, slider=w):
                    _set_nested(self.runtime_params, p, slider.value())
                w.valueChanged.connect(_on_val)
            elif kind == COMBO:
                def _on_text(_t, p=path):
                    _set_nested(self.runtime_params, p, _t)
                w.currentTextChanged.connect(_on_text)
            elif kind == CHECK:
                def _on_check(_s, p=path):
                    _set_nested(self.runtime_params, p, bool(_s))
                w.stateChanged.connect(_on_check)
            elif kind == RADIO_F0:
                w.toggled.connect(lambda _: self._on_f0_method_changed())
            elif kind == RADIO_SR:
                w.toggled.connect(lambda _: self._on_sr_mode_changed())

        # 2. RangeSlider（音域映射）— rangeChanged 信号
        if hasattr(self, "exp_pitch_map_src_range"):
            def _on_src_range(low, high):
                self.runtime_params.voice.pitch_map_src_min = float(low)
                self.runtime_params.voice.pitch_map_src_max = float(high)
            self.exp_pitch_map_src_range.rangeChanged.connect(_on_src_range)

        if hasattr(self, "exp_pitch_map_dst_range"):
            def _on_dst_range(low, high):
                self.runtime_params.voice.pitch_map_dst_min = float(low)
                self.runtime_params.voice.pitch_map_dst_max = float(high)
            self.exp_pitch_map_dst_range.rangeChanged.connect(_on_dst_range)

        # 3. F0 阈值滑动条 — 根据当前 F0 方法更新对应字段
        # 注意：DoubleSlider 的 valueChanged 发射内部编码整数值，必须读取 slider.value()
        if hasattr(self, "exp_f0_threshold_slider"):
            def _on_f0_threshold(_v, slider=self.exp_f0_threshold_slider):
                val = slider.value()
                if self.runtime_params.f0.method == "rmvpe":
                    self.runtime_params.f0.rmvpe_threshold = val
                else:
                    self.runtime_params.f0.fcpe_confidence_threshold = val
            self.exp_f0_threshold_slider.valueChanged.connect(_on_f0_threshold)

    def _on_sr_mode_changed(self):
        """采样率模式切换时，更新 runtime_params.audio.sr_mode。"""
        self.runtime_params.audio.sr_mode = "model" if self.sr_model_radio.isChecked() else "device"



    def _on_f0_method_changed(self):
        """F0 方法切换时，更新 runtime_params 和阈值滑动条。"""
        is_rmvpe = self.f0_rmvp_btn.isChecked()
        self.runtime_params.f0.method = "rmvpe" if is_rmvpe else "fcpe"
        if hasattr(self, "exp_f0_threshold_slider") and hasattr(self, "exp_f0_threshold_name"):
            params = self.runtime_params
            val = params.f0.rmvpe_threshold if is_rmvpe else params.f0.fcpe_confidence_threshold
            self.exp_f0_threshold_slider.setValue(val)
            self.exp_f0_threshold_name.setText("RMVPE 阈值" if is_rmvpe else "FCPE 阈值")



    def _update_timer(self):

        if self.engine.running and self.engine.measure_ms > 0:

            self.delay_lbl.setText(f"延迟: {self.engine.measure_ms:.0f}ms")

            # 显示当前输入音高（从 pipeline state 获取，音域映射之前的原始值）
            try:
                input_pitch = self.engine._runner.pipeline.state.last_input_pitch
            except Exception:
                input_pitch = 0.0
            if input_pitch > 0:
                self.pitch_lbl.setText(f"音高: {input_pitch:.0f}Hz")
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



    def _apply_runtime_params(self):
        """全量同步所有参数到 runtime_params（启动时和点击开始时调用）。

        信号连接已实时同步，这里做一次全量原地更新确保一致性。
        使用 update_from 原地复制字段，不替换对象引用，
        engine/runner/pipeline 持有的引用自动生效。
        """
        state = self.collect_gui_state()
        self.runtime_params.update_from(state)



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



    def collect_gui_state(self) -> InferenceParams:

        return collect_params(self)



    def apply_gui_state(self, state: InferenceParams) -> None:

        apply_params(self, state)



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

            stats = self.controller.setup_engine(

                sr_mode=state.audio.sr_mode,

                input_device_idx=self.device_manager.get_input_device_index(self.input_combo.currentIndex()),

                output_device_idx=self.device_manager.get_output_device_index(self.output_combo.currentIndex()),

                output2_device_idx=self.device_manager.get_output_device_index(self.output2_combo.currentIndex() - 1),

                block_time=state.buffer.block_time,

                crossfade_time=state.buffer.crossfade_time,

                extra_time=state.buffer.extra_time,

                enable_out2=state.audio.enable_out2,

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

        self._show_error(str(e))



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







