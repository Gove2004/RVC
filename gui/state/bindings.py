"""GUI 控件 ↔ InferenceParams 字段绑定。

双向通路：BINDINGS 表驱动 apply_params（配置加载 → 控件回填）与
collect_params（保存配置 → 从控件收集），collect/apply 都在本模块。

新增参数时改动点：
1. rvc/core/config.py 对应分组 dataclass 加字段
2. BINDINGS 加一行（分组路径 + 控件 + 读写方式）
3. Tab 里建控件，创建时传入 params 初始值

默认值单源：所有缺省值只活在 InferenceParams dataclass 上，
params_from_dict 从「刚构造出的默认实例」取 fallback，不写第二份字面量。
"""
from typing import TYPE_CHECKING

from rvc.core.config import PARAM_GROUPS, InferenceParams

if TYPE_CHECKING:
    from gui.view.contracts import InferWindowHost

# 读写方式
FLOAT = "float"
COMBO = "combo"
PATH = "path"
RADIO_F0 = "radio_f0"
RADIO_SR = "radio_sr"

# 序列化分组名与派生字段（params_from_dict / params_to_dict / 测试共用单源）
GROUP_NAMES = PARAM_GROUPS  # 单源：rvc.core.config
# 派生属性，不参与 dict 序列化（enable_out2 由 output2_device 计算）
DERIVED_FIELDS = frozenset({"enable_out2"})
# 顶层字段
TOP_LEVEL_FIELDS = ("rms_mix", "volume", "model_path", "hubert")

# 状态字段 schema：(分组路径, window 控件属性名, 读写方式)
# 分组路径用点号分隔，如 "voice.formant"、"buffer.block_time"
BINDINGS = [
    # ── 音色参数（voice.*）──
    ("voice.formant", "formant_slider", FLOAT),
    ("rms_mix", "rms_mix_slider", FLOAT),
    ("volume", "volume_slider", FLOAT),
    # ── F0 参数（f0.*）──
    ("f0.method", "f0_rmvpe_btn", RADIO_F0),
    # ── 缓冲区参数（buffer.*）──
    ("buffer.block_time", "block_time_slider", FLOAT),
    ("buffer.crossfade_time", "crossfade_slider", FLOAT),
    ("buffer.extra_time", "extra_time_slider", FLOAT),
    # ── 音频设备参数（audio.*）──
    ("audio.sr_mode", "sr_model_radio", RADIO_SR),
    ("audio.input_device", "input_combo", COMBO),
    ("audio.output_device", "output_combo", COMBO),
    ("audio.output2_device", "output2_combo", COMBO),
    # ── 顶层 ──
    # 模型路径由 PathPickerButton 自身持有（path()），不另存平行字段
    ("model_path", "model_path_btn", PATH),
    ("hubert", "hubert_combo", COMBO),
]


def _get_nested(obj, path: str):
    """按点号路径递归读取嵌套字段。"""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_nested(obj, path: str, value):
    """按点号路径递归写入嵌套字段。"""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ── 派生字段（唯一事实源）──
# 副输出「不使用」是 output2_combo 的首项占位文本（重填见 MainWindow.device_populate）
OUTPUT2_NONE_TEXT = "不使用"

# F0 阈值标签文案单源（apply_params / 窗口切换方法两处消费）
F0_THRESHOLD_LABELS = {"rmvpe": "RMVPE 阈值", "fcpe": "FCPE 阈值"}


def f0_threshold_label(method: str) -> str:
    """当前 F0 方法对应的阈值滑条标签文案。"""
    return F0_THRESHOLD_LABELS.get(method, F0_THRESHOLD_LABELS["fcpe"])


def f0_active_threshold(params: InferenceParams) -> float:
    """当前 F0 方法对应的阈值（滑条应展示的值）。"""
    if params.f0.method == "rmvpe":
        return params.f0.rmvpe_threshold
    return params.f0.fcpe_confidence_threshold


def f0_store_threshold(params: InferenceParams, value: float) -> None:
    """把阈值滑条值写入当前 F0 方法对应的字段。"""
    if params.f0.method == "rmvpe":
        params.f0.rmvpe_threshold = value
    else:
        params.f0.fcpe_confidence_threshold = value


def output2_enabled(output2_device: str) -> bool:
    """副输出是否启用 = output2_device 的派生（下拉框选到真实设备才算启用）。"""
    return bool(output2_device) and output2_device != OUTPUT2_NONE_TEXT


def params_from_dict(state: dict) -> InferenceParams:
    """从分组 dict 构造 InferenceParams。

    动态遍历各分组的 __dataclass_fields__，新字段自动包含，无需手动更新。
    缺省值取 InferenceParams 默认（单源）：`params` 是刚构造的默认实例，
    每个字段只被 setattr 一次，所以读 `getattr(group, field)` 拿到的仍是
    dataclass 默认值——不需要再单独持有一份 _DEFAULTS 副本（那样反而多了
    一个跨模块共享的可变实例）。
    特殊字段：enable_out2 是 output2_device 的派生属性，device_index 转 int。
    """
    params = InferenceParams()

    for group_name in GROUP_NAMES:
        group_data = state.get(group_name, {})
        group = getattr(params, group_name)
        for field_name in group.__dataclass_fields__:
            if field_name in DERIVED_FIELDS:
                continue
            setattr(group, field_name, group_data.get(field_name, getattr(group, field_name)))

    # 派生属性：enable_out2 从 output2_device 计算
    params.audio.enable_out2 = output2_enabled(params.audio.output2_device)
    # device_index 强制 int（JSON 读出来可能是 float）
    params.audio.device_index = int(params.audio.device_index)

    # 顶层字段（rms_mix/volume/model_path/hubert）
    for field_name in TOP_LEVEL_FIELDS:
        setattr(params, field_name, state.get(field_name, getattr(params, field_name)))
    return params


def params_to_dict(params: InferenceParams) -> dict:
    """把 InferenceParams 序列化为分组 dict（动态遍历，新字段自动包含）。

    派生字段 enable_out2 不持久化（从 output2_device 计算）。
    """
    result = {}

    for group_name in GROUP_NAMES:
        group = getattr(params, group_name)
        result[group_name] = {
            field_name: getattr(group, field_name)
            for field_name in group.__dataclass_fields__
            if field_name not in DERIVED_FIELDS
        }

    for field_name in TOP_LEVEL_FIELDS:
        result[field_name] = getattr(params, field_name)
    return result


def _set(win: "InferWindowHost", widget, kind, value):
    """把值写到控件。"""
    if kind == FLOAT:
        slider = getattr(win, widget)
        slider.setValue(float(value))
        # 手动刷新标签：控件未显示时 setValue 可能不触发 valueChanged
        slider.update_label()
    elif kind == COMBO:
        combo = getattr(win, widget)
        # 优先按 userData 匹配，无 userData 时回退到文本匹配
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)
    elif kind == PATH:
        # 程序化回填（配置加载）：只改显示与内部路径，不反向触发 pathChanged
        getattr(win, widget).set_path(str(value) if value else "", emit=False)
    elif kind == RADIO_F0:
        win.f0_rmvpe_btn.setChecked(value == "rmvpe")
        win.f0_fcpe_btn.setChecked(value != "rmvpe")
    elif kind == RADIO_SR:
        win.sr_model_radio.setChecked(value == "model")
        win.sr_device_radio.setChecked(value != "model")
    else:
        raise ValueError(f"未知读写方式: {kind}")


def apply_params(win: "InferWindowHost", params: InferenceParams) -> None:
    """把 InferenceParams 写到所有控件（启动时加载配置调用）。

    所有 BINDINGS 控件与实验控件都在窗口构建时无条件创建，不做 hasattr 探测——
    控件缺失应立刻 AttributeError 暴露，而不是静默跳过。
    """
    for path, widget, kind in BINDINGS:
        value = _get_nested(params, path)
        _set(win, widget, kind, value)

    # F0 阈值滑动条
    slider = win.f0_threshold_slider
    slider.setValue(f0_active_threshold(params))
    slider.update_label()
    win.f0_threshold_name.setText(f0_threshold_label(params.f0.method))

    # 音域映射 RangeSlider
    win.pitch_map_src_range.set_selected_range(
        float(params.voice.pitch_map_src_min), float(params.voice.pitch_map_src_max)
    )
    win.pitch_map_dst_range.set_selected_range(
        float(params.voice.pitch_map_dst_min), float(params.voice.pitch_map_dst_max)
    )


def collect_params(win: "InferWindowHost") -> InferenceParams:
    """从控件收集 InferenceParams（保存配置时调用）。"""
    params = InferenceParams()
    for path, widget, kind in BINDINGS:
        w = getattr(win, widget)
        if kind == FLOAT:
            value = w.value()
        elif kind == COMBO:
            # 优先取 userData，无则取文本
            data = w.currentData()
            value = data if data is not None else w.currentText()
        elif kind == PATH:
            value = w.path()
        elif kind == RADIO_F0:
            value = "rmvpe" if win.f0_rmvpe_btn.isChecked() else "fcpe"
        elif kind == RADIO_SR:
            value = "model" if win.sr_model_radio.isChecked() else "device"
        else:
            continue
        _set_nested(params, path, value)

    # F0 阈值：滑条只展示当前方法的阈值；非活动方法的阈值从 runtime_params
    # 回填——若另一方法落回 dataclass 默认值，任何一次保存都会重置用户调好的值
    val = float(win.f0_threshold_slider.value())
    f0_store_threshold(params, val)
    if params.f0.method == "rmvpe":
        params.f0.fcpe_confidence_threshold = win.runtime_params.f0.fcpe_confidence_threshold
    else:
        params.f0.rmvpe_threshold = win.runtime_params.f0.rmvpe_threshold

    params.voice.pitch_map_src_min = float(win.pitch_map_src_range.low())
    params.voice.pitch_map_src_max = float(win.pitch_map_src_range.high())
    params.voice.pitch_map_dst_min = float(win.pitch_map_dst_range.low())
    params.voice.pitch_map_dst_max = float(win.pitch_map_dst_range.high())

    # enable_out2 是 output2_device 的派生属性，根据当前选择动态计算
    params.audio.enable_out2 = output2_enabled(params.audio.output2_device)

    # GPU 编号：没有走 BINDINGS（下拉项来自运行时探测的显卡列表），
    # 其变化由 gpu_combo 的信号直接写入 runtime_params，故这里取实时值。
    params.audio.device_index = int(getattr(win, "runtime_params").audio.device_index)

    return params
