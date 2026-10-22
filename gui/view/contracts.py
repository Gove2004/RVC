"""View 宿主契约 — 推理主窗口对生命周期混入与跨对象消费方暴露的结构化协议。

以 Protocol 显式声明消费方通过宿主访问的全部属性与方法，替代「隐式宿主属性」。
结构化契约：`MainWindow` 满足全部协议；
`WindowLifecycle`/`bindings`/`tabs`/`OfflineConversion`/`DeviceCatalog` 以此为宿主类型。
运行期零使用（仅类型层）——本模块不承载任何行为。

接口隔离：每个消费方只声明自己用到的窄面（`OfflineViewHost`/`DeviceViewHost`
不继承 `InferWindowHost`），controller 层因此永远看不到控件——widget 知识
只存在于 view 层（window.py / tabs / lifecycle）。
"""
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PySide6.QtCore import QTimer, Signal
    from PySide6.QtWidgets import (
        QComboBox, QLabel, QLineEdit, QProgressBar, QPushButton,
        QRadioButton, QSlider, QTextEdit,
    )

    from gui.controller.device_controller import DeviceCatalog
    from gui.controller.engine_controller import InferController
    from gui.controller.offline_controller import OfflineConversion
    from gui.controller.tool_controller import ModelToolController, ToolRunner
    from gui.view.tray import SystemTray
    from gui.view.widgets import (
        DoubleSlider, ElidedLabel, LoadThread, ModeToggle, PathPickerButton, RangeSlider,
    )
    from rvc.core.config import InferenceParams
    from rvc.streaming.engine import VoiceEngine


class InferWindowHost(Protocol):
    """`MainWindow` 的宿主面（生命周期混入与参数绑定的唯一事实源）。

    各属性均在 `MainWindow.__init__` / tab 构建函数中无条件创建；
    控件缺失应立刻 AttributeError 暴露，而不是静默跳过。
    """

    # ── 核心协作对象 ──
    controller: "InferController"
    runtime_params: "InferenceParams"
    device_catalog: "DeviceCatalog"
    tray: "SystemTray | None"

    # ── 模型工具（融合/详情页签 + 生命周期退出等待）──
    model_tools: "ModelToolController"
    tool_runner: "ToolRunner | None"

    # ── 生命周期状态（window.__init__ 初始化）──
    _timer: "QTimer"
    _lt: "LoadThread | None"
    _loading: bool
    # 遥测 tooltip 变更缓存（内容不变不重写，避免每 tick 触发重绘）
    _last_delay_tooltip: str
    _last_pitch_text: str | None
    _last_error_count: int

    # ── 离线转换（生命周期与启动互斥都会碰它）──
    offline_conversion: "OfflineConversion"

    # ── 音色控件（timbre_tab）──
    model_path_btn: "PathPickerButton"
    hubert_combo: "QComboBox"
    pitch_map_src_range: "RangeSlider"
    pitch_map_dst_range: "RangeSlider"
    formant_slider: "DoubleSlider"
    rms_mix_slider: "DoubleSlider"
    volume_slider: "DoubleSlider"

    # ── 参数控件（params_tab）──
    block_time_slider: "DoubleSlider"
    extra_time_slider: "DoubleSlider"
    crossfade_slider: "DoubleSlider"
    f0_threshold_slider: "DoubleSlider"
    f0_threshold_name: "QLabel"
    f0_rmvpe_btn: "QRadioButton"
    f0_fcpe_btn: "QRadioButton"

    # ── 设备控件（device_tab / BINDINGS）──
    hostapi_combo: "QComboBox"
    input_combo: "QComboBox"
    output_combo: "QComboBox"
    output2_combo: "QComboBox"
    gpu_combo: "QComboBox"
    sr_model_radio: "QRadioButton"
    sr_device_radio: "QRadioButton"

    # ── 融合控件（merge_tab）──
    merge_a: "PathPickerButton"
    merge_b: "PathPickerButton"
    merge_slider: "QSlider"
    merge_label: "QLabel"
    merge_name: "QLineEdit"
    btn_merge: "QPushButton"
    merge_status: "QLabel"

    # ── 详情控件（details_tab）──
    inspect_path: "PathPickerButton"
    inspect_result: "QTextEdit"

    # ── 离线控件（offline_tab）──
    offline_input_btn: "PathPickerButton"
    offline_output: "ElidedLabel"
    offline_button: "QPushButton"
    offline_preview_btn: "QPushButton"
    offline_delete_btn: "QPushButton"
    offline_status: "QLabel"
    offline_progress: "QProgressBar"

    # ── 底部控制栏 ──
    mode_toggle: "ModeToggle"

    # ── 底部状态栏控件 ──
    delay_lbl: "QLabel"
    pitch_lbl: "QLabel"
    error_lbl: "QLabel"

    # ── 信号（音频回调线程 → 主线程错误转发）──
    runtime_error: "Signal"
    # 预热线程 → 主线程：显卡列表回填 / 预热失败提示
    gpu_list_ready: "Signal"
    warmup_error: "Signal"

    # ── 惰性引擎 ──
    @property
    def engine(self) -> "VoiceEngine": ...

    # ── 宿主方法（window.py 定义，混入与绑定层调用）──
    def apply_gui_state(self, state: "InferenceParams") -> None: ...

    def collect_gui_state(self) -> "InferenceParams": ...

    def _apply_runtime_params(self) -> None: ...

    def _mark_loading(self) -> None: ...

    def _mark_running(self) -> None: ...

    def _reset_runtime_ui(self) -> None: ...

    def show_error(self, message: str) -> None: ...

    def show_warning(self, message: str) -> None: ...


class OfflineViewHost(Protocol):
    """离线转换的视图面 — `OfflineConversion` 的唯一 UI 依赖。

    controller 只消费语义（选了什么、进入什么状态），控件机械操作
    （置灰/文案/进度条/对话框）全部由 MainWindow 在 view 层完成。
    """

    # ── 核心协作对象（互斥守卫与引擎获取）──
    controller: "InferController"

    # ── 读取（view 从控件归结；路径已 strip）──
    def offline_input_path(self) -> str: ...

    def offline_model_path(self) -> str: ...

    def offline_hubert_variant(self) -> str: ...

    def is_engine_loading(self) -> bool:
        """模型是否正在后台加载（离线启动互斥用：加载中 is_running 仍为 False）。"""
        ...

    def collect_gui_state(self) -> "InferenceParams": ...

    # ── 反馈 ──
    def show_warning(self, message: str) -> None: ...

    def show_error(self, message: str) -> None: ...

    def offline_confirm_delete(self, path: str) -> bool:
        """删除输出文件的二次确认（弹窗在 view 层）。"""
        ...

    # ── 转换态 UI（三态 + 进度/状态行）──
    def offline_enter_converting(self) -> None:
        """进入转换态：主按钮改「取消」，监听/删除置灰，状态「初始化...」，进度清零。"""
        ...

    def offline_enter_cancelling(self) -> None: ...

    def offline_reset(self, status: str) -> None:
        """转换结束（完成/取消/出错）复位按钮组。"""
        ...

    def offline_set_progress(self, current: int, total: int) -> None: ...

    def offline_set_status(self, text: str) -> None: ...

    def offline_set_previewing(self, on: bool, name: str = "") -> None:
        """监听按钮文案（「停止」/「监听」）与状态行；name 为播放中的文件名。"""
        ...


class DeviceViewHost(Protocol):
    """设备目录的视图面 — `DeviceCatalog` 的唯一 UI 依赖。

    controller 提供枚举数据与「保留选择」的决策，重填/屏蔽信号/占位项等
    控件机械操作由 view 完成。
    """

    def device_current_selections(self) -> tuple[str, str, str, str]:
        """四个下拉框的当前文本：(hostapi, 输入, 输出, 副输出)。"""
        ...

    def device_populate(
        self,
        hostapis: list[str],
        inputs: list[str],
        outputs: list[str],
        sel: tuple[str, str, str, str],
    ) -> None:
        """重填四个下拉框并恢复 sel 中的选择（hostapi 重填期间屏蔽信号防递归；
        副输出首项恒为占位「不使用」）。"""
        ...
