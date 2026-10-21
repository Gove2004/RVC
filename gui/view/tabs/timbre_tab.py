"""音色 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子 / 气息感"""
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox

from gui.view.widgets import _create_slider_row, RangeSlider, PathPickerButton

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost


def _range_row(min_val, max_val, step, low_val, high_val,
               fmt=".0f", unit="Hz", label_w=80):
    """创建「双滑块范围 + 自动格式化值标签」，返回 (range_slider, label)。"""
    rs = RangeSlider(min_val, max_val, step, low_val, high_val, fmt=fmt, unit=unit)
    lbl = QLabel()
    lbl.setFixedWidth(label_w)
    lbl.setAlignment(Qt.AlignCenter)

    def _fmt(low, high):
        return f"{low:{fmt}}-{high:{fmt}}{unit}"

    lbl.setText(_fmt(rs.low(), rs.high()))
    rs.rangeChanged.connect(lambda low, high: lbl.setText(_fmt(low, high)))
    return rs, lbl


def build_timbre_tab(win: "InferWindowHost"):
    """音色 Tab — 模型路径 / 音域映射 / 性别因子 / 响度因子 / 气息感。"""
    params = win.runtime_params
    w = QWidget()
    g = QGridLayout(w)
    g.setSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    g.setColumnStretch(0, 4)
    g.setColumnStretch(1, 13)
    g.setColumnStretch(2, 3)
    r = 0

    # ── 模型路径 + 特征器 ──
    # 路径由按钮自带（path()），不再另存 win.model_path 平行字段；
    # 参数绑定里 model_path 走 PATH 方式直接读按钮。
    win.model_path_btn = PathPickerButton(
        win, title="选择模型", filter_str="模型 (*.pth)",
        placeholder="选择模型...",
        base_dir=lambda: win.controller.models_dir,
    )

    win.hubert_combo = QComboBox()
    win.hubert_combo.addItems(["base", "chinese"])
    win.hubert_combo.setMinimumHeight(28)
    win.hubert_combo.setToolTip("此模型训练时用的特征器（base=原版 hubert_base，chinese=腾讯中文 hubert）。训练与推理必须一致。")
    win.hubert_combo.setMaximumWidth(90)

    g.addWidget(QLabel("模型路径"), r, 0)
    g.addWidget(win.model_path_btn, r, 1)
    g.addWidget(win.hubert_combo, r, 2)
    r += 1

    # ── 原声音域 ──
    win.pitch_map_src_range, pitch_map_src_label = _range_row(
        10.0, 1000.0, 10.0,
        params.voice.pitch_map_src_min, params.voice.pitch_map_src_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("原声音域"), r, 0)
    g.addWidget(win.pitch_map_src_range, r, 1)
    g.addWidget(pitch_map_src_label, r, 2)
    r += 1

    # ── 模型音域 ──
    win.pitch_map_dst_range, pitch_map_dst_label = _range_row(
        10.0, 1000.0, 10.0,
        params.voice.pitch_map_dst_min, params.voice.pitch_map_dst_max,
        fmt=".0f", unit="Hz",
    )
    g.addWidget(QLabel("模型音域"), r, 0)
    g.addWidget(win.pitch_map_dst_range, r, 1)
    g.addWidget(pitch_map_dst_label, r, 2)
    r += 1

    # ── 性别因子（formant）──
    win.formant_slider, formant_label = _create_slider_row(-2.5, 2.5, 0.05, 0.0, fmt="+.2f")
    g.addWidget(QLabel("性别因子"), r, 0)
    g.addWidget(win.formant_slider, r, 1)
    g.addWidget(formant_label, r, 2)
    r += 1

    # ── 响度因子（rms_mix）──
    win.rms_mix_slider, rms_mix_label = _create_slider_row(0.0, 1.0, 0.05, 0.0)
    g.addWidget(QLabel("响度因子"), r, 0)
    g.addWidget(win.rms_mix_slider, r, 1)
    g.addWidget(rms_mix_label, r, 2)
    r += 1

    # ── 气息感（texture.airflow：说话时呼呼声 + 静音底噪）──
    win.airflow_slider, airflow_label = _create_slider_row(0.0, 1.0, 0.05, 0.0)
    g.addWidget(QLabel("气息感"), r, 0)
    g.addWidget(win.airflow_slider, r, 1)
    g.addWidget(airflow_label, r, 2)
    r += 1

    # ── 检索强度（index_rate：HuBERT 特征按索引近邻混合的比例）──
    # 需要 <模型名>.index.pt 特征索引文件（与 pth 同目录），缺失时自动无效
    win.index_rate_slider, index_rate_label = _create_slider_row(0.0, 1.0, 0.05, 0.0)
    win.index_rate_slider.setToolTip(
        "特征检索强度：用训练素材的特征索引改写 HuBERT 特征（音色更像训练目标）。\n"
        "需要模型同目录的 <模型名>.index.pt 索引文件（python -m rvc.train.index_builder 构建），\n"
        "缺失时本滑块无效。改动需重启变声生效。"
    )
    g.addWidget(QLabel("检索强度"), r, 0)
    g.addWidget(win.index_rate_slider, r, 1)
    g.addWidget(index_rate_label, r, 2)
    r += 1

    # ── 辅音保护（protect：清音帧特征混回检索前版本，防检索抹掉清辅音细节）──
    # 特征侧混合（原版 RVC 语义）：合成器仍只渲染目标音色，100% 也不漏原声。
    # 仅在检索强度>0 时有意义。
    win.protect_slider, protect_label = _create_slider_row(0.0, 1.0, 0.05, 0.0)
    win.protect_slider.setToolTip(
        "辅音保护：清音帧（辅音）特征按比例混回检索前的原始特征，防止检索改写清辅音细节。\n"
        "特征侧混合，不漏原声；只在「检索强度」>0 时有意义。改动需重启变声生效。"
    )
    g.addWidget(QLabel("辅音保护"), r, 0)
    g.addWidget(win.protect_slider, r, 1)
    g.addWidget(protect_label, r, 2)
    r += 1

    return w
