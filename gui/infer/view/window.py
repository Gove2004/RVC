"""推理 GUI 主窗口 — View 层：窗口装配 + 信号绑定（D5 拆分三职责之一/之二）。

GUI 分层后职责划分：
- 本文件：UI 构建（tabs/控制栏）、参数控件信号绑定、UI 状态更新（按钮文案/标签）、
  委托业务逻辑给 InferController
- lifecycle.py：生命周期（配置持久化、预热、启动/停止、托盘退出、关闭行为）

业务逻辑（参数应用、引擎控制、错误处理）由 InferController 承担。
"""
import logging
import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTabWidget, QSpacerItem, QSizePolicy,
)

from PySide6.QtCore import QTimer, Signal

from gui.infer.controller.engine_controller import InferController

from gui.infer.state.bindings import (
    collect_params,
    apply_params,
)

from gui.infer.view.lifecycle import WindowLifecycle
from gui.infer.view.widgets import LoadThread

from gui.infer.view.tabs.audio_driver_tab import build_audio_driver_tab

from gui.infer.view.tabs.global_params_tab import build_timbre_tab

from gui.infer.view.tabs.offline_tab import build_offline_tab

from gui.infer.view.tabs.experimental_tab import build_performance_tab

from gui.infer.controller.device_controller import DeviceCatalog

from gui.infer.controller.offline_controller import OfflineConversion

from gui.styles import ButtonStyles, Layout
from gui.styles.colors import Colors

if TYPE_CHECKING:
    # D5 例外：类型注解可以引用 rvc 类型（运行时不 import）
    from rvc.core.config import InferenceParams

logger = logging.getLogger(__name__)


class MainWindow(WindowLifecycle, QMainWindow):
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
        self.device_catalog = DeviceCatalog(self)
        self.offline_conversion = OfflineConversion(self)

        self.device_catalog.load_hostapis()
        self._load_gui_config()
        # Connect refresh button after device catalog is ready
        self.refresh_btn.clicked.connect(self._reload_dev)
        # 窗口稳定后后台预热引擎（torch 加载 ~1.6s），避免首次点「开始」卡顿
        QTimer.singleShot(300, self._warmup_engine)

        # 系统托盘：关闭=最小化到托盘；托盘不可用时直接报错退出
        self.tray = None
        try:
            from gui.infer.view.tray import SystemTray

            idle_icon, active_icon = self.controller.tray_icon_paths
            self.tray = SystemTray(self, on_quit=self._tray_quit,
                                    icon_idle=idle_icon, icon_active=active_icon)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"托盘初始化失败：{e}\n程序将退出。")
            sys.exit(1)

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

    # ── 窗口装配 ──

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
        tabs.addTab(build_timbre_tab(self), "音色调节")
        tabs.addTab(build_performance_tab(self), "性能调节")
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

        # 错误计数：实时链路吞错（清零输出继续跑）的唯一可见信号，>0 才显示
        self.error_lbl = QLabel("")
        self.error_lbl.setMinimumWidth(70)
        self.error_lbl.setStyleSheet(f"color: {Colors.ERROR};")
        self.error_lbl.hide()
        btn_group.addWidget(self.error_lbl)

        ctrl.addLayout(btn_group)
        root.addLayout(ctrl)

    # ── 信号绑定 ──

    def _connect_all_param_signals(self) -> None:
        """统一连接所有参数控件的变化信号 → 实时更新 runtime_params。

        覆盖 BINDINGS 表中的所有控件类型（FLOAT/COMBO/RADIO_F0/RADIO_SR），
        以及特殊控件（RangeSlider、f0_threshold_slider）。
        使用 _signals_connected 标志确保只连接一次，避免重复连接。
        """
        if getattr(self, '_signals_connected', False):
            return
        self._signals_connected = True

        from gui.infer.state.bindings import (
            BINDINGS, FLOAT, COMBO, RADIO_F0, RADIO_SR, _set_nested,
        )

        # 1. BINDINGS 表中的所有控件
        for path, widget, kind in BINDINGS:
            if not widget or not hasattr(self, widget):
                continue
            w = getattr(self, widget)
            if kind == FLOAT:
                # 注意：DoubleSlider 的 valueChanged 发射内部编码整数值，
                # 所以必须读取 w.value() 获取物理值，而不是用信号参数
                def _on_val(_v, p=path, slider=w):
                    _set_nested(self.runtime_params, p, slider.value())
                w.valueChanged.connect(_on_val)
            elif kind == COMBO:
                def _on_text(_t, p=path):
                    _set_nested(self.runtime_params, p, _t)
                w.currentTextChanged.connect(_on_text)
            elif kind == RADIO_F0:
                w.toggled.connect(lambda _: self._on_f0_method_changed())
            elif kind == RADIO_SR:
                w.toggled.connect(lambda _: self._on_sr_mode_changed())

        # 2. RangeSlider（音域映射）— rangeChanged 信号
        if hasattr(self, "pitch_map_src_range"):
            def _on_src_range(low, high):
                self.runtime_params.voice.pitch_map_src_min = float(low)
                self.runtime_params.voice.pitch_map_src_max = float(high)
            self.pitch_map_src_range.rangeChanged.connect(_on_src_range)

        if hasattr(self, "pitch_map_dst_range"):
            def _on_dst_range(low, high):
                self.runtime_params.voice.pitch_map_dst_min = float(low)
                self.runtime_params.voice.pitch_map_dst_max = float(high)
            self.pitch_map_dst_range.rangeChanged.connect(_on_dst_range)

        # 3. F0 阈值滑动条 — 根据当前 F0 方法更新对应字段
        # 注意：DoubleSlider 的 valueChanged 发射内部编码整数值，必须读取 slider.value()
        if hasattr(self, "f0_threshold_slider"):
            def _on_f0_threshold(_v, slider=self.f0_threshold_slider):
                val = slider.value()
                if self.runtime_params.f0.method == "rmvpe":
                    self.runtime_params.f0.rmvpe_threshold = val
                else:
                    self.runtime_params.f0.fcpe_confidence_threshold = val
            self.f0_threshold_slider.valueChanged.connect(_on_f0_threshold)

    def _on_sr_mode_changed(self):
        """采样率模式切换时，更新 runtime_params.audio.sr_mode。"""
        self.runtime_params.audio.sr_mode = "model" if self.sr_model_radio.isChecked() else "device"

    def _on_f0_method_changed(self):
        """F0 方法切换时，更新 runtime_params 和阈值滑动条。"""
        is_rmvpe = self.f0_rmvp_btn.isChecked()
        self.runtime_params.f0.method = "rmvpe" if is_rmvpe else "fcpe"
        params = self.runtime_params
        val = params.f0.rmvpe_threshold if is_rmvpe else params.f0.fcpe_confidence_threshold
        self.f0_threshold_slider.setValue(val)
        self.f0_threshold_name.setText("RMVPE 阈值" if is_rmvpe else "FCPE 阈值")

    # ── 设备/离线委托 ──

    def _reload_dev(self):
        """委托给 DeviceCatalog（运行中禁止刷新，防止杀活动流）"""
        if self.controller.is_running:
            self._show_warning("运行中不能刷新设备，请先停止")
            return
        self.device_catalog.reload_devices()

    def _ha_changed(self, name):
        self.device_catalog.on_hostapi_changed(name)

    def _off_browse(self, tgt, kind):
        self.offline_conversion.browse_file(tgt, kind)

    def _off_start(self):
        # 保存配置（在开始转换前保存当前设置）
        try:
            self._save_gui_config()
            logger.debug("配置已保存")
        except Exception as e:
            logger.warning("保存配置失败：%s", e)
        self.offline_conversion.start_conversion()

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
        self.error_lbl.hide()

    def _mark_loading(self):
        self._set_toggle_button("加载中", False, ButtonStyles.secondary())

    def _mark_running(self):
        self._set_toggle_button("停止", True, ButtonStyles.danger())
        self._timer.start(200)

    def collect_gui_state(self) -> "InferenceParams":
        return collect_params(self)

    def apply_gui_state(self, state: "InferenceParams") -> None:
        apply_params(self, state)
