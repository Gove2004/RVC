"""性能调节 Tab — 缓冲区参数 / 音高算法 / F0 阈值"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QRadioButton, QButtonGroup,
)

from gui.infer.view.widgets import _create_slider_row


def build_performance_tab(win):
    """性能调节 Tab — 采样长度 / 上下文 / 淡入 / 音高算法 / F0 阈值。"""
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

    # ── 淡入长度（crossfade_time；上限 0.04 = SOLA 实际生效的 40ms，再大无效）──
    win.crossfade_slider, crossfade_label = _create_slider_row(0.01, 0.04, 0.01, 0.04)
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

    # ── F0 阈值（根据当前选择的 F0 方法动态切换）──
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

    return w
