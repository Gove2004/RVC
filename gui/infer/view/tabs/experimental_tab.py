"""实验功能 Tab — 辅音保护 / 破音保护"""
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout,
)

from gui.infer.view.widgets import _slrow, _sl_value_as_float
from rvc.core.experimental import experimental_config


def _group_with_toggle(title: str, checkbox: QCheckBox) -> tuple[QGroupBox, QGridLayout]:
    """创建带标题+启用勾选框的 GroupBox，勾选框在标题文字右侧。"""
    box = QGroupBox()
    outer = QVBoxLayout(box)
    outer.setContentsMargins(8, 4, 8, 8)

    header = QHBoxLayout()
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-weight: bold;")
    header.addWidget(title_lbl)
    header.addWidget(checkbox)
    header.addStretch()
    outer.addLayout(header)

    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    outer.addLayout(grid)
    return box, grid


def build_experimental_tab(win):
    w = QWidget()
    root = QVBoxLayout(w)

    # ── 1. 辅音保护 ──
    win.exp_protect_soft_checkbox = QCheckBox("启用")
    win.exp_protect_soft_checkbox.setChecked(experimental_config.protect_soft_enabled)
    win.exp_protect_soft_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "protect_soft_enabled", win.exp_protect_soft_checkbox.isChecked())
    )
    group1, g1 = _group_with_toggle("辅音保护", win.exp_protect_soft_checkbox)
    r = 0

    # 保护强度
    win.protect_slider = _slrow(win, "protect_slider", 0.0, 1.0, 0.01, 0.5)
    g1.addWidget(QLabel("保护强度"), r, 0)
    g1.addWidget(win.protect_slider, r, 1)
    g1.addWidget(win.protect_label, r, 2); r += 1

    # 过渡中心
    win.exp_protect_threshold_slider = _slrow(
        win, "exp_protect_threshold_slider", 0.0, 30.0, 1.0,
        experimental_config.protect_soft_threshold_hz, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_threshold_hz",
            _sl_value_as_float(win.exp_protect_threshold_slider),
        )
    )
    g1.addWidget(QLabel("过渡中心"), r, 0)
    g1.addWidget(win.exp_protect_threshold_slider, r, 1)
    g1.addWidget(win.exp_protect_threshold_label, r, 2); r += 1

    # 过渡宽度
    win.exp_protect_width_slider = _slrow(
        win, "exp_protect_width_slider", 5.0, 50.0, 1.0,
        experimental_config.protect_soft_width, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_protect_width_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "protect_soft_width",
            _sl_value_as_float(win.exp_protect_width_slider),
        )
    )
    g1.addWidget(QLabel("过渡宽度"), r, 0)
    g1.addWidget(win.exp_protect_width_slider, r, 1)
    g1.addWidget(win.exp_protect_width_label, r, 2); r += 1

    root.addWidget(group1)

    # ── 2. 半音尺度音域映射（实验功能，默认关闭） ──
    win.exp_pitch_map_checkbox = QCheckBox("启用")
    win.exp_pitch_map_checkbox.setChecked(experimental_config.pitch_map_enabled)
    win.exp_pitch_map_checkbox.stateChanged.connect(
        lambda: setattr(experimental_config, "pitch_map_enabled", win.exp_pitch_map_checkbox.isChecked())
    )
    group_pm, gpm = _group_with_toggle("音域映射（半音尺度，保持音程）", win.exp_pitch_map_checkbox)
    r = 0

    # 音调大小（全局 pitch，关闭音域映射时作为固定半音偏移生效）
    win.pitch_slider = _slrow(win, "pitch_slider", -16, 16, 1, 0, fmt="+0.0f")
    gpm.addWidget(QLabel("音调大小"), r, 0)
    gpm.addWidget(win.pitch_slider, r, 1)
    gpm.addWidget(win.pitch_label, r, 2); r += 1

    # 原声音域下限
    win.exp_pitch_map_src_min_slider = _slrow(
        win, "exp_pitch_map_src_min_slider", 50.0, 300.0, 5.0,
        experimental_config.pitch_map_src_min, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_pitch_map_src_min_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "pitch_map_src_min",
            _sl_value_as_float(win.exp_pitch_map_src_min_slider),
        )
    )
    gpm.addWidget(QLabel("原声音域 下限"), r, 0)
    gpm.addWidget(win.exp_pitch_map_src_min_slider, r, 1)
    gpm.addWidget(win.exp_pitch_map_src_min_label, r, 2); r += 1

    # 原声音域上限
    win.exp_pitch_map_src_max_slider = _slrow(
        win, "exp_pitch_map_src_max_slider", 100.0, 500.0, 5.0,
        experimental_config.pitch_map_src_max, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_pitch_map_src_max_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "pitch_map_src_max",
            _sl_value_as_float(win.exp_pitch_map_src_max_slider),
        )
    )
    gpm.addWidget(QLabel("原声音域 上限"), r, 0)
    gpm.addWidget(win.exp_pitch_map_src_max_slider, r, 1)
    gpm.addWidget(win.exp_pitch_map_src_max_label, r, 2); r += 1

    # 目标音域下限
    win.exp_pitch_map_dst_min_slider = _slrow(
        win, "exp_pitch_map_dst_min_slider", 100.0, 500.0, 5.0,
        experimental_config.pitch_map_dst_min, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_pitch_map_dst_min_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "pitch_map_dst_min",
            _sl_value_as_float(win.exp_pitch_map_dst_min_slider),
        )
    )
    gpm.addWidget(QLabel("目标音域 下限"), r, 0)
    gpm.addWidget(win.exp_pitch_map_dst_min_slider, r, 1)
    gpm.addWidget(win.exp_pitch_map_dst_min_label, r, 2); r += 1

    # 目标音域上限
    win.exp_pitch_map_dst_max_slider = _slrow(
        win, "exp_pitch_map_dst_max_slider", 200.0, 800.0, 5.0,
        experimental_config.pitch_map_dst_max, fmt=".0f", unit="Hz", label_w=45,
    )
    win.exp_pitch_map_dst_max_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "pitch_map_dst_max",
            _sl_value_as_float(win.exp_pitch_map_dst_max_slider),
        )
    )
    gpm.addWidget(QLabel("目标音域 上限"), r, 0)
    gpm.addWidget(win.exp_pitch_map_dst_max_slider, r, 1)
    gpm.addWidget(win.exp_pitch_map_dst_max_label, r, 2); r += 1

    root.addWidget(group_pm)

    # ── 3. F0 清浊阈值 ──
    win.exp_rmvpe_threshold_slider = _slrow(
        win, "exp_rmvpe_threshold_slider", 0.01, 0.10, 0.01,
        experimental_config.rmvpe_threshold, fmt=".2f", label_w=45,
    )
    win.exp_rmvpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "rmvpe_threshold",
            _sl_value_as_float(win.exp_rmvpe_threshold_slider),
        )
    )
    group3 = QGroupBox("F0 清浊阈值")
    g3 = QGridLayout(group3)
    g3.setHorizontalSpacing(8)
    r = 0
    g3.addWidget(QLabel("RMVPE 阈值"), r, 0)
    g3.addWidget(win.exp_rmvpe_threshold_slider, r, 1)
    g3.addWidget(win.exp_rmvpe_threshold_label, r, 2); r += 1

    win.exp_fcpe_threshold_slider = _slrow(
        win, "exp_fcpe_threshold_slider", 0.01, 0.08, 0.01,
        experimental_config.fcpe_confidence_threshold, fmt=".2f", label_w=45,
    )
    win.exp_fcpe_threshold_slider.valueChanged.connect(
        lambda: setattr(
            experimental_config, "fcpe_confidence_threshold",
            _sl_value_as_float(win.exp_fcpe_threshold_slider),
        )
    )
    g3.addWidget(QLabel("FCPE 阈值"), r, 0)
    g3.addWidget(win.exp_fcpe_threshold_slider, r, 1)
    g3.addWidget(win.exp_fcpe_threshold_label, r, 2); r += 1
    root.addWidget(group3)

    root.addStretch()
    return w
