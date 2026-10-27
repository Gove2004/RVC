"""参数 Tab — 块时长 / 上下文长度 / 交叉淡化 / 音高算法 / 音高阈值 / 真实感"""
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QHBoxLayout, QLabel, QRadioButton, QButtonGroup,
    QSizePolicy,
)

from gui.view.widgets import create_slider_row

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost


def build_params_tab(win: "InferWindowHost"):
    """参数 Tab — 块时长 / 上下文长度 / 交叉淡化 / 音高算法 / 音高阈值 / 真实感。"""
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
    win.block_time_slider, block_time_label = create_slider_row(0.05, 0.50, 0.01, 0.25)
    g.addWidget(QLabel("块时长"), r, 0)
    g.addWidget(win.block_time_slider, r, 1)
    g.addWidget(block_time_label, r, 2)
    r += 1

    # ── 上下文长度（extra_time）──
    win.extra_time_slider, extra_time_label = create_slider_row(0.10, 5.0, 0.10, 2.5)
    g.addWidget(QLabel("上下文长度"), r, 0)
    g.addWidget(win.extra_time_slider, r, 1)
    g.addWidget(extra_time_label, r, 2)
    r += 1

    # ── 交叉淡化（crossfade_time）──
    win.crossfade_slider, crossfade_label = create_slider_row(0.01, 0.05, 0.01, 0.05)
    g.addWidget(QLabel("交叉淡化"), r, 0)
    g.addWidget(win.crossfade_slider, r, 1)
    g.addWidget(crossfade_label, r, 2)
    r += 1

    # ── 音高算法 ──
    # 两枚单选钮必须包进容器再入格（同设备页采样率行）：裸放进拉伸列时，
    # 其较小的 sizeHint/最大宽会把整列的 max 钳死，且两钮被拆到两列显得散；
    # 容器 + 尾部 addStretch 让富余宽度进弹性空隙、两钮贴文字宽度横排靠左。
    win.f0_rmvpe_btn = QRadioButton("RMVPE")
    win.f0_fcpe_btn = QRadioButton("FCPE")
    is_rmvpe = params.f0.method == "rmvpe"
    win.f0_rmvpe_btn.setChecked(is_rmvpe)
    win.f0_fcpe_btn.setChecked(not is_rmvpe)

    f0_group = QButtonGroup(w)
    f0_group.addButton(win.f0_rmvpe_btn)
    f0_group.addButton(win.f0_fcpe_btn)

    f0_row = QWidget()
    # 垂直必须 Fixed：容器默认 Preferred（可拉伸），而本页其余行都是 Fixed，
    # 窗口的富余高度会被这一行独吞，radio 垂直居中后上下留白、行明显偏高。
    f0_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    f0_h = QHBoxLayout(f0_row)
    f0_h.setContentsMargins(0, 0, 0, 0)
    f0_h.setSpacing(12)
    f0_h.addWidget(win.f0_rmvpe_btn)
    f0_h.addWidget(win.f0_fcpe_btn)
    f0_h.addStretch(1)

    g.addWidget(QLabel("音高算法"), r, 0)
    g.addWidget(f0_row, r, 1, 1, 2)
    r += 1

    # ── 音高阈值（行标签随音高算法动态切换为 RMVPE/FCPE 阈值）──
    initial_threshold = params.f0.rmvpe_threshold if params.f0.method == "rmvpe" else params.f0.fcpe_confidence_threshold
    win.f0_threshold_slider, f0_threshold_label = create_slider_row(
        0.01, 0.10, 0.01,
        initial_threshold, fmt=".2f",
    )
    win.f0_threshold_name = QLabel("RMVPE 阈值" if params.f0.method == "rmvpe" else "FCPE 阈值")

    g.addWidget(win.f0_threshold_name, r, 0)
    g.addWidget(win.f0_threshold_slider, r, 1)
    g.addWidget(f0_threshold_label, r, 2)
    r += 1

    return w
