"""参数 Tab — 采样长度 / 上下文长度 / 淡入长度 / 音高算法 / 音高阈值 / 输出音量"""
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QRadioButton, QButtonGroup,
)

from gui.view.widgets import _create_slider_row

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost


def build_params_tab(win: "InferWindowHost"):
    """参数 Tab — 采样长度 / 上下文长度 / 淡入长度 / 音高算法 / 音高阈值 / 输出音量。"""
    params = win.runtime_params
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 采样长度（block_time）──
    win.block_time_slider, block_time_label = _create_slider_row(0.05, 0.50, 0.01, 0.25)
    g.addWidget(QLabel("采样长度"), r, 0)
    g.addWidget(win.block_time_slider, r, 1)
    g.addWidget(block_time_label, r, 2)
    r += 1

    # ── 上下文长度（extra_time）──
    win.extra_time_slider, extra_time_label = _create_slider_row(0.10, 5.0, 0.10, 2.5)
    g.addWidget(QLabel("上下文长度"), r, 0)
    g.addWidget(win.extra_time_slider, r, 1)
    g.addWidget(extra_time_label, r, 2)
    r += 1

    # ── 淡入长度（crossfade_time）──
    win.crossfade_slider, crossfade_label = _create_slider_row(0.01, 0.05, 0.01, 0.05)
    g.addWidget(QLabel("淡入长度"), r, 0)
    g.addWidget(win.crossfade_slider, r, 1)
    g.addWidget(crossfade_label, r, 2)
    r += 1

    # ── 音高算法 ──
    win.f0_rmvp_btn = QRadioButton("RMVPE")
    win.f0_rmvp_btn.setMaximumWidth(80)
    win.f0_fcpe_btn = QRadioButton("FCPE")
    win.f0_fcpe_btn.setMaximumWidth(80)
    is_rmvpe = params.f0.method == "rmvpe"
    win.f0_rmvp_btn.setChecked(is_rmvpe)
    win.f0_fcpe_btn.setChecked(not is_rmvpe)

    f0_group = QButtonGroup(w)
    f0_group.addButton(win.f0_rmvp_btn)
    f0_group.addButton(win.f0_fcpe_btn)

    g.addWidget(QLabel("音高算法"), r, 0)
    g.addWidget(win.f0_rmvp_btn, r, 1)
    g.addWidget(win.f0_fcpe_btn, r, 2)
    r += 1

    # ── 音高阈值（行标签随音高算法动态切换为 RMVPE/FCPE 阈值）──
    initial_threshold = params.f0.rmvpe_threshold if params.f0.method == "rmvpe" else params.f0.fcpe_confidence_threshold
    win.f0_threshold_slider, f0_threshold_label = _create_slider_row(
        0.01, 0.10, 0.01,
        initial_threshold, fmt=".2f",
    )
    win.f0_threshold_name = QLabel("RMVPE 阈值" if params.f0.method == "rmvpe" else "FCPE 阈值")

    g.addWidget(win.f0_threshold_name, r, 0)
    g.addWidget(win.f0_threshold_slider, r, 1)
    g.addWidget(f0_threshold_label, r, 2)
    r += 1

    # ── 辅音保护（protect；0=不保护，1=清音帧完全用原声）──
    win.protect_slider, protect_label = _create_slider_row(
        0.0, 1.0, 0.05, params.protect, fmt=".2f",
    )
    protect_slider = win.protect_slider
    protect_slider.setToolTip(
        "辅音保护：F0 低于「辅音阈值」的帧按比例混入原始麦克风音频，\n"
        "防止「嘶/施/次」等清辅音被转换成电音或撕裂。\n"
        "0=关闭；越大清辅音越接近原声（咬字更真实，但音色一致性下降）。"
    )
    g.addWidget(QLabel("辅音保护"), r, 0)
    g.addWidget(protect_slider, r, 1)
    g.addWidget(protect_label, r, 2)
    r += 1

    # ── 辅音阈值（protect_threshold_hz；F0 低于此值视为清音帧）──
    win.protect_threshold_slider, protect_threshold_label = _create_slider_row(
        0.0, 50.0, 5.0, params.protect_threshold_hz, fmt=".0f", unit="Hz",
    )
    protect_threshold_slider = win.protect_threshold_slider
    protect_threshold_slider.setToolTip(
        "清浊判定阈值：连续 F0 低于此值的帧按「辅音保护」比例混入原声。\n"
        "RMVPE/FCPE 对清辅音输出接近 0 的 F0，默认 25Hz 只捕获真清音；\n"
        "调大则更多浊音帧也被保护（咬字更真，音色一致性更差）。"
    )
    g.addWidget(QLabel("辅音阈值"), r, 0)
    g.addWidget(protect_threshold_slider, r, 1)
    g.addWidget(protect_threshold_label, r, 2)
    r += 1

    return w
