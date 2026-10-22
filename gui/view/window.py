"""推理 GUI 主窗口 — View 层：窗口装配 + 信号绑定。

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
    QHBoxLayout, QLabel, QMessageBox,
    QTabWidget, QSpacerItem, QSizePolicy,
)

from PySide6.QtCore import QTimer, Signal

from gui.controller.engine_controller import InferController

from gui.state.bindings import (
    collect_params,
    apply_params,
    f0_active_threshold,
    f0_store_threshold,
    f0_threshold_label,
)

from gui.view.lifecycle import WindowLifecycle

from gui.view.tabs.device_tab import build_device_tab

from gui.view.tabs.timbre_tab import build_timbre_tab

from gui.view.tabs.params_tab import build_params_tab

from gui.view.tabs.offline_tab import build_offline_tab

from gui.view.tabs.merge_tab import build_merge_tab

from gui.view.tabs.details_tab import build_details_tab

from gui.controller.device_controller import DeviceCatalog

from gui.controller.offline_controller import OfflineConversion

from gui.controller.tool_controller import ModelToolController

from gui.styles import Layout
from gui.styles.colors import Colors
from gui.state.bindings import OUTPUT2_NONE_TEXT

from gui.view.widgets import ModeToggle

if TYPE_CHECKING:
    # 类型注解可引用 rvc 类型（仅 TYPE_CHECKING，运行时不 import）
    from rvc.core.config import InferenceParams

logger = logging.getLogger(__name__)


class MainWindow(WindowLifecycle, QMainWindow):
    """推理主窗口 — 实现 View 宿主契约 `gui.view.contracts.InferWindowHost`。

    装配顺序：UI 构建（含各 tab 控件挂载）→ 信号绑定 → 管理器初始化 →
    配置加载 → 托盘。生命周期行为在 `WindowLifecycle` 混入中。
    """

    # 音频回调线程产生的运行时错误通过此信号转发到主线程（禁止回调线程操作 Qt）
    runtime_error = Signal(str)
    # 引擎预热线程探测到的显卡列表 → 主线程回填「显卡」下拉框
    gpu_list_ready = Signal(list)
    # 预热线程的初始化失败 → 主线程弹错误框（后台线程不能直接操作 Qt，
    # 也不能用无 context 的 QTimer.singleShot——非 Qt 线程上定时器回调永不执行）
    warmup_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC")
        self.resize(200, 150)
        self.controller = InferController(on_runtime_error=self._on_runtime_error)
        self.runtime_params = self.controller.runtime_params
        self.runtime_error.connect(self._handle_runtime_error)
        self.warmup_error.connect(self.show_error)
        self.gpu_list_ready.connect(self._populate_gpu_combo)
        self._loading = False
        self._lt = None
        self._last_delay_tooltip = None
        # 遥测文本缓存：内容未变时跳过 setText/show/hide（每 200ms 一跳）
        self._last_pitch_text = None
        self._last_error_count = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        # 管理器先于 UI 创建：offline_tab 构建时会连接 offline_conversion 的信号
        self.offline_conversion = OfflineConversion(self)
        self._build_ui()
        self._connect_all_param_signals()

        # 初始化管理器
        self.device_catalog = DeviceCatalog(self)
        self.model_tools = ModelToolController()
        self.tool_runner = None
        self.tool_queue = []

        # 同步加载设备（PortAudio ~170ms 在窗口出现前完成）：
        # 先填下拉框再载入配置，_load_gui_config 的 COMBO 回填才能直接命中
        self.device_catalog.load_hostapis()
        self._load_gui_config()
        # 初始三档按钮 UI（停止档）
        self._refresh_mode()
        # device_catalog 就绪后再连接刷新按钮信号
        self.refresh_btn.clicked.connect(self._reload_dev)
        # 窗口事件循环启动后立即后台预热引擎（torch 加载 ~1.7s，后台线程不抢主线程；
        # 预热越早完成，点「开始」越不容易撞上预热）
        QTimer.singleShot(0, self._warmup_engine)

        # 系统托盘：关闭=最小化到托盘；托盘不可用时直接报错退出
        self.tray = None
        try:
            from gui.view.tray import SystemTray

            idle_icon, active_icon, pass_icon = self.controller.tray_icon_paths
            self.tray = SystemTray(self, on_quit=self._tray_quit,
                                    icon_idle=idle_icon, icon_active=active_icon,
                                    icon_passthrough=pass_icon)
            # 按实际运行状态刷新一次三态图标（初始为停止档红色）
            self.tray.update_status()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"托盘初始化失败：{e}\n程序将退出。")
            sys.exit(1)

    @property
    def engine(self):
        """惰性获取引擎 — 首次访问才构造（构造会加载 torch，避免拖慢窗口出现）。"""
        return self.controller.engine

    # ── UI 辅助方法 ──

    def show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "提示", message)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "错误", message)

    # ── 控制器契约实现（离线 OfflineViewHost / 设备 DeviceViewHost）──
    # controller 层禁止触碰控件；这里把「语义」归结为「控件机械操作」。
    # 按钮文案恒 2 字（换长度会撑动整列宽度，刻意设计）。

    def offline_input_path(self) -> str:
        return self.offline_input_btn.path().strip()

    def offline_model_path(self) -> str:
        return self.model_path_btn.path().strip()

    def offline_hubert_variant(self) -> str:
        return self.hubert_combo.currentText()

    def is_engine_loading(self) -> bool:
        """模型是否正在后台加载（生命周期混入维护 _loading）。"""
        return self._loading

    def offline_confirm_delete(self, path: str) -> bool:
        ans = QMessageBox.question(
            self, "删除输出文件",
            f"确定删除该输出文件吗？\n{path}\n\n文件将移入回收站。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def offline_enter_converting(self) -> None:
        self.offline_button.setText("取消")
        self.offline_preview_btn.setEnabled(False)
        self.offline_delete_btn.setEnabled(False)
        self.offline_status.setText("初始化...")
        self.offline_progress.setValue(0)

    def offline_enter_cancelling(self) -> None:
        self.offline_button.setEnabled(False)
        self.offline_status.setText("正在取消...")

    def offline_reset(self, status: str) -> None:
        self.offline_button.setEnabled(True)
        self.offline_button.setText("开始")
        self.offline_preview_btn.setEnabled(True)
        self.offline_delete_btn.setEnabled(True)
        self.offline_status.setText(status)

    def offline_set_progress(self, current: int, total: int) -> None:
        self.offline_status.setText(f"进度: {current}/{total}")
        if total > 0:
            self.offline_progress.setValue(current)

    def offline_set_status(self, text: str) -> None:
        self.offline_status.setText(text)

    def offline_set_previewing(self, on: bool, name: str = "") -> None:
        self.offline_preview_btn.setText("停止" if on else "监听")
        if on:
            self.offline_status.setText(f"监听中: {name}")

    def device_current_selections(self) -> tuple[str, str, str, str]:
        return (
            self.hostapi_combo.currentText(),
            self.input_combo.currentText(),
            self.output_combo.currentText(),
            self.output2_combo.currentText(),
        )

    def device_populate(self, hostapis, ins, outs, sel) -> None:
        """重填四个设备下拉框并恢复原选择。

        四个下拉框全部屏蔽信号后再重填：currentTextChanged 会触发
        _ha_changed 递归刷新（PortAudio 全量枚举跑两遍）；副输出重填期间的
        clear() 还会让 _on_output2_changed 把 enable_out2 瞬时写成 False
        （设备不在新列表时甚至会永久脱节）。派生字段在重填后统一同步一次。
        副输出首项恒为占位「不使用」（单源 bindings）。
        """
        ha, inp, outp, out2 = sel
        combos = (self.hostapi_combo, self.input_combo,
                  self.output_combo, self.output2_combo)
        for combo in combos:
            combo.blockSignals(True)
        try:
            self.hostapi_combo.clear()
            self.hostapi_combo.addItems(hostapis)
            self._select_combo(self.hostapi_combo, ha)

            self.input_combo.clear()
            self.input_combo.addItems(ins)
            self.output_combo.clear()
            self.output_combo.addItems(outs)
            self.output2_combo.clear()
            self.output2_combo.addItem(OUTPUT2_NONE_TEXT)
            self.output2_combo.addItems(outs)
            for combo, text in ((self.input_combo, inp),
                                (self.output_combo, outp),
                                (self.output2_combo, out2)):
                self._select_combo(combo, text)
            # 信号屏蔽期间不会触发，这里补一次派生同步（也保证恢复失败时
            # 派生字段如实反映「不使用」）
            self._on_output2_changed(self.output2_combo.currentText())
        finally:
            for combo in combos:
                combo.blockSignals(False)

    @staticmethod
    def _select_combo(combo, text: str) -> None:
        """若 text 仍在列表中则恢复选中，否则保持默认首项。"""
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ── 窗口装配 ──

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()

        # 设备页签同时返回刷新按钮（device_catalog 就绪后由 window 连信号）
        device_w, self.refresh_btn = build_device_tab(self)
        tabs.addTab(device_w, "设备")
        tabs.addTab(build_timbre_tab(self), "音色")
        tabs.addTab(build_params_tab(self), "参数")
        tabs.addTab(build_offline_tab(self), "离线")
        tabs.addTab(build_merge_tab(self), "融合")
        tabs.addTab(build_details_tab(self), "详情")
        root.addWidget(tabs)

        # ── 底部控制栏 ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(Layout.SPACING_SMALL)

        # 左侧：三档运行态 toggle（原声/停止/变声，分段互斥单选）——
        # 控件只发 modeChanged，档位执行由 lifecycle 分发；UI 勾选态由
        # _refresh_mode 按实际流状态回写
        btn_group = QHBoxLayout()
        self.mode_toggle = ModeToggle()
        self.mode_toggle.modeChanged.connect(self._on_mode_changed)
        btn_group.addWidget(self.mode_toggle)

        spacer1 = QSpacerItem(40, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        btn_group.addSpacerItem(spacer1)

        # 右侧：输入音高 + 延迟显示
        self.pitch_lbl = QLabel("音高: -")
        self.pitch_lbl.setMinimumWidth(78)
        self.pitch_lbl.setToolTip("当前输入音高（Hz，实时检测）：辅助调节高级功能中的音域映射参数")
        btn_group.addWidget(self.pitch_lbl)

        self.delay_lbl = QLabel("延迟: -")
        self.delay_lbl.setMinimumWidth(88)
        self.delay_lbl.setToolTip("端到端延迟（含输入/输出驱动缓冲）。鼠标悬停查看分解。想降延迟调小「采样长度」，或换用低延迟音频驱动")
        btn_group.addWidget(self.delay_lbl)

        # 错误计数：实时链路吞错（清零输出继续跑）的唯一可见信号，>0 才显示
        self.error_lbl = QLabel("")
        self.error_lbl.setMinimumWidth(56)
        self.error_lbl.setStyleSheet(f"color: {Colors.ERROR};")
        self.error_lbl.hide()
        btn_group.addWidget(self.error_lbl)

        ctrl.addLayout(btn_group)
        root.addLayout(ctrl)

    # ── 信号绑定 ──

    def _connect_all_param_signals(self) -> None:
        """统一连接所有参数控件的变化信号 → 实时更新 runtime_params。

        覆盖 BINDINGS 表中的所有控件类型（FLOAT/COMBO/RADIO_F0/RADIO_SR），
        以及特殊控件（RangeSlider、f0_threshold_slider）。只在 __init__ 调用一次。
        """
        from gui.state.bindings import (
            BINDINGS, FLOAT, COMBO, PATH, RADIO_F0, RADIO_SR, _set_nested,
        )

        # 1. BINDINGS 表中的所有控件
        for path, widget, kind in BINDINGS:
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
            elif kind == PATH:
                def _on_path(new_path, p=path):
                    _set_nested(self.runtime_params, p, new_path)
                w.pathChanged.connect(_on_path)
            elif kind == RADIO_F0:
                w.toggled.connect(lambda _: self._on_f0_method_changed())
            elif kind == RADIO_SR:
                w.toggled.connect(lambda _: self._on_sr_mode_changed())

        # 2. RangeSlider（音域映射）— rangeChanged 信号
        def _on_src_range(low, high):
            self.runtime_params.voice.pitch_map_src_min = float(low)
            self.runtime_params.voice.pitch_map_src_max = float(high)
        self.pitch_map_src_range.rangeChanged.connect(_on_src_range)

        def _on_dst_range(low, high):
            self.runtime_params.voice.pitch_map_dst_min = float(low)
            self.runtime_params.voice.pitch_map_dst_max = float(high)
        self.pitch_map_dst_range.rangeChanged.connect(_on_dst_range)

        # 3. F0 阈值滑动条 — 根据当前 F0 方法更新对应字段
        # 注意：DoubleSlider 的 valueChanged 发射内部编码整数值，必须读取 slider.value()
        def _on_f0_threshold(_v, slider=self.f0_threshold_slider):
            f0_store_threshold(self.runtime_params, slider.value())
        self.f0_threshold_slider.valueChanged.connect(_on_f0_threshold)

        # 4. 缓冲区参数滑块（块时长/交叉淡化/额外上下文）— 运行中改动自动停止转换。
        #    这三个值在启动时一次性决定缓冲区分配（runner.init_processing），
        #    运行中改动对当前流不生效；为避免「改了没反应」的错觉，运行中拖动
        #    即自动停止，新值在下次启动生效（_on_loaded → setup_engine 重读滑块）。
        #    加载中不停止：_on_loaded 会重新 collect_gui_state，拖动的新值自然被采用。
        #    配置加载/启动回填也会触发 valueChanged，靠 is_running 守卫过滤（此时为 False）。
        for _slider_name in ("block_time_slider", "crossfade_slider", "extra_time_slider"):
            _slider = getattr(self, _slider_name)

            def _on_buffer_param_changed(_v, s=_slider):
                if self.controller.is_running:
                    logger.info("缓冲区参数（块时长/交叉淡化/额外上下文）变更，自动停止转换（新值下次启动生效）")
                    self._stop()

            _slider.valueChanged.connect(_on_buffer_param_changed)

        # 5. 副输出下拉框 — enable_out2 是 output2_device 的派生值，
        #    BINDINGS 的通用处理器只写 output2_device，这里补上派生字段，
        #    否则「选了副输出设备但没重启过」时 collect_params 之外的地方读到旧值。
        self.output2_combo.currentTextChanged.connect(self._on_output2_changed)

        # 6. 显卡下拉框 — 直接写 runtime_params.audio.device_index（无 BINDINGS 路径，
        #    下拉项来自运行时探测的显卡列表）。
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_changed)

    def _on_output2_changed(self, text: str) -> None:
        """副输出选择变化时同步派生字段 enable_out2。"""
        from gui.state.bindings import output2_enabled

        self.runtime_params.audio.enable_out2 = output2_enabled(text)

    def _on_gpu_changed(self, index: int) -> None:
        """显卡选择变化时写入 runtime_params（重启后经 RuntimeDeviceConfig 生效）。"""
        if index >= 0:
            self.runtime_params.audio.device_index = int(index)

    def _populate_gpu_combo(self, names: list) -> None:
        """回填显卡下拉框（引擎预热线程探测完成后经信号投递到主线程）。

        names 为空（无 CUDA / 探测失败）时保持占位项不动。
        """
        if not names:
            return
        combo = self.gpu_combo
        combo.blockSignals(True)
        combo.clear()
        for i, name in enumerate(names):
            combo.addItem(f"GPU {i}: {name}")
        index = int(self.runtime_params.audio.device_index)
        if 0 <= index < len(names):
            combo.setCurrentIndex(index)
        else:
            # 越界（换卡/删卡）回落到 GPU 0，并且**同步写回状态**：
            # 回填全程屏蔽信号，_on_gpu_changed 不会触发；只改 UI 会让
            # runtime_params 里的旧索引一直留着，下次保存配置就又写回文件。
            logger.warning("配置的 GPU 编号 %d 超出本机显卡数 %d，回落到 GPU 0", index, len(names))
            combo.setCurrentIndex(0)
            self.runtime_params.audio.device_index = 0
        combo.blockSignals(False)

    def _on_sr_mode_changed(self):
        """采样率模式切换时，更新 runtime_params.audio.sr_mode。"""
        self.runtime_params.audio.sr_mode = "model" if self.sr_model_radio.isChecked() else "device"

    def _on_f0_method_changed(self):
        """F0 方法切换时，更新 runtime_params 和阈值滑动条。"""
        is_rmvpe = self.f0_rmvpe_btn.isChecked()
        self.runtime_params.f0.method = "rmvpe" if is_rmvpe else "fcpe"
        self.f0_threshold_slider.setValue(f0_active_threshold(self.runtime_params))
        self.f0_threshold_name.setText(f0_threshold_label(self.runtime_params.f0.method))

    # ── 设备/离线委托 ──

    def _reload_dev(self):
        """委托给 DeviceCatalog（运行中禁止刷新，防止杀活动流）"""
        if self.controller.is_running or self.controller.passthrough_running:
            self.show_warning("运行中不能刷新设备，请先停止")
            return
        self.device_catalog.reload_devices()

    def _ha_changed(self, name):
        """音频驱动切换（下拉框直选）— 与刷新按钮是同一条「全量重扫+重填」链路。

        必须与 _reload_dev 同守卫：运行中重扫会把四个下拉框清空重填，期间
        output2 的 clear() 会让派生字段 enable_out2 瞬时变 False；若副输出设备
        不在新驱动的列表里，恢复失败后该字段就永久脱节，随后任意一次保存都会
        把脱节状态持久化。拒绝时把下拉框回退到实际生效的驱动（不重扫）。
        """
        if self.controller.is_running or self.controller.passthrough_running:
            self.show_warning("运行中不能切换音频驱动，请先停止")
            applied = self.device_catalog.applied_hostapi
            self.hostapi_combo.blockSignals(True)
            self._select_combo(self.hostapi_combo, applied)
            self.hostapi_combo.blockSignals(False)
            return
        self.device_catalog.on_hostapi_changed(name)

    def _toggle_offline(self):
        """离线 tab 的开始/取消单按钮调度（空闲时保存配置并开始）。"""
        if self.offline_conversion.is_converting:
            self.offline_conversion.cancel_conversion()
            return
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

    def _refresh_mode(self):
        """按实际运行状态收敛三档 toggle UI（唯一事实源 = 实际流状态）。"""
        if self._loading:
            mode = "loading"
        elif self.controller.is_running:
            mode = "convert"
        elif self.controller.passthrough_running:
            mode = "passthrough"
        else:
            mode = "stop"
        self.mode_toggle.set_mode(mode)

    def _reset_runtime_ui(self):
        self._timer.stop()
        # 文本与「上次写过的文本」必须一起复位：_update_timer 的变更守卫会
        # 比较这两个值，只改文本不清缓存的话，停止→再启动后首个 tick 若算出
        # 与停止前相同的值就会被判「没变」而跳过写盘，界面永久停在「-」。
        self._last_delay_tooltip = None
        self._last_pitch_text = "音高: -"
        self._last_error_count = 0
        self.delay_lbl.setText("延迟: -")
        self.pitch_lbl.setText("音高: -")
        self.error_lbl.hide()
        self._refresh_mode()

    def _mark_loading(self):
        """加载中：三档 toggle 落到加载态；遥测定时器停掉（流还没起）。"""
        self._timer.stop()
        self._refresh_mode()

    def _mark_running(self):
        self._timer.start(200)
        self._refresh_mode()

    def collect_gui_state(self) -> "InferenceParams":
        return collect_params(self)

    def apply_gui_state(self, state: "InferenceParams") -> None:
        apply_params(self, state)
