"""参数 Tab — 音高算法 / 音高阈值 / 块时长 / 上下文长度 / 交叉淡化。"""
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox

from gui.state import bindings
from gui.view.widgets import create_slider_row

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost


def build_params_tab(win: "InferWindowHost"):
    """参数 Tab — 音高算法 / 音高阈值 / 块时长 / 上下文长度 / 交叉淡化。"""
    params = win.runtime_params
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 音高算法 ──
    win.f0_combo = QComboBox()
    win.f0_combo.addItem("RMVPE", "rmvpe")
    win.f0_combo.addItem("FCPE", "fcpe")
    win.f0_combo.setMinimumHeight(28)
    win.f0_combo.setCurrentIndex(0 if params.f0.method == "rmvpe" else 1)
    g.addWidget(QLabel("音高算法"), r, 0)
    g.addWidget(win.f0_combo, r, 1, 1, 2)
    r += 1

    # ── 音高阈值（行标签随音高算法动态切换为 RMVPE/FCPE 阈值）──
    # 阈值/标签文案统一走 bindings 单源 helper（与 apply_params/方法切换共用，防漂移）
    win.f0_threshold_slider, f0_threshold_label = create_slider_row(
        0.01, 0.10, 0.01,
        bindings.f0_active_threshold(params), fmt=".2f",
    )
    win.f0_threshold_name = QLabel(bindings.f0_threshold_label(params.f0.method))

    g.addWidget(win.f0_threshold_name, r, 0)
    g.addWidget(win.f0_threshold_slider, r, 1)
    g.addWidget(f0_threshold_label, r, 2)
    r += 1

    # ── 采样长度（block_time）──
    win.block_time_slider, block_time_label = create_slider_row(0.05, 0.25, 0.01, params.buffer.block_time)
    g.addWidget(QLabel("块时长"), r, 0)
    g.addWidget(win.block_time_slider, r, 1)
    g.addWidget(block_time_label, r, 2)
    r += 1

    # ── 上下文长度（extra_time）──
    win.extra_time_slider, extra_time_label = create_slider_row(0.10, 5.0, 0.10, params.buffer.extra_time)
    g.addWidget(QLabel("上下文长度"), r, 0)
    g.addWidget(win.extra_time_slider, r, 1)
    g.addWidget(extra_time_label, r, 2)
    r += 1

    # ── 交叉淡化（crossfade_time；上限 0.10 = SOLA 实际生效的 100ms）──
    win.crossfade_slider, crossfade_label = create_slider_row(0.01, 0.10, 0.01, params.buffer.crossfade_time)
    g.addWidget(QLabel("交叉淡化"), r, 0)
    g.addWidget(win.crossfade_slider, r, 1)
    g.addWidget(crossfade_label, r, 2)
    r += 1

    return w
