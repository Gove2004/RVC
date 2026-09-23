"""View 宿主契约 — 推理主窗口对生命周期混入与跨对象消费方暴露的结构化协议。

§④ C-14 / D-10：以 Protocol 显式声明 `WindowLifecycle` 混入通过 `self` 访问的
全部宿主属性与方法，替代「mixin 隐式宿主属性」。结构化契约：`MainWindow` 满足之；
`WindowLifecycle`/`bindings`/`global_params_tab`/`experimental_tab` 以此为宿主类型。
运行期零使用（仅类型层）——本模块不承载任何行为。
"""
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PySide6.QtCore import QTimer, Signal
    from PySide6.QtWidgets import (
        QComboBox, QLabel, QPushButton, QRadioButton,
    )

    from gui.infer.controller.device_controller import DeviceCatalog
    from gui.infer.controller.engine_controller import InferController
    from gui.infer.view.tray import SystemTray
    from gui.infer.view.widgets import DoubleSlider, LoadThread, RangeSlider
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

    # ── 生命周期状态（window.__init__ 初始化）──
    _timer: "QTimer"
    _lt: "LoadThread | None"
    _loading: bool

    # ── 音色调节控件（global_params_tab）──
    model_path: str
    model_path_btn: "QPushButton"
    hubert_combo: "QComboBox"
    pitch_map_src_range: "RangeSlider"
    pitch_map_dst_range: "RangeSlider"
    formant_slider: "DoubleSlider"
    rms_mix_slider: "DoubleSlider"

    # ── 性能调节控件（experimental_tab）──
    block_time_slider: "DoubleSlider"
    extra_time_slider: "DoubleSlider"
    crossfade_slider: "DoubleSlider"
    f0_threshold_slider: "DoubleSlider"
    f0_threshold_name: "QLabel"
    f0_rmvp_btn: "QRadioButton"
    f0_fcpe_btn: "QRadioButton"

    # ── 设备驱动控件（audio_driver_tab / BINDINGS）──
    hostapi_combo: "QComboBox"
    input_combo: "QComboBox"
    output_combo: "QComboBox"
    output2_combo: "QComboBox"
    sr_model_radio: "QRadioButton"
    sr_device_radio: "QRadioButton"

    # ── 底部状态栏控件 ──
    delay_lbl: "QLabel"
    pitch_lbl: "QLabel"
    error_lbl: "QLabel"

    # ── 信号（音频回调线程 → 主线程错误转发）──
    runtime_error: "Signal"

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

    def _show_error(self, message: str) -> None: ...

    def _show_warning(self, message: str) -> None: ...
