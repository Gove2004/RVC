"""GUI 控件 ↔ InferenceParams 字段绑定。

滑动条直接绑定 InferenceParams 的分组字段，
valueChanged 时自动更新参数对象，不再需要 collect/apply 双向搬运。

新增参数时改动点：
1. rvc/core/config.py 对应分组 dataclass 加字段
2. BINDINGS 加一行（分组路径 + 控件 + 读写方式 + 默认值）
3. Tab 里建控件，创建时传入 params 初始值
"""
from rvc.core.config import InferenceParams

# 读写方式
CHECK = "check"
FLOAT = "float"
INT = "int"
COMBO = "combo"
TEXT = "text"
ATTR = "attr"
RADIO_F0 = "radio_f0"
RADIO_SR = "radio_sr"

# 状态字段 schema：(分组路径, window 控件属性名, 读写方式, 缺省默认值)
# 分组路径用点号分隔，如 "voice.formant"、"buffer.block_time"
BINDINGS = [
    # ── 音色参数（voice.*）──
    ("voice.formant", "formant_slider", FLOAT, 0.0),
    ("rms_mix", "rms_mix_slider", FLOAT, 0.0),
    # ── F0 参数（f0.*）──
    ("f0.method", "f0_rmvp_btn", RADIO_F0, "rmvpe"),
    # ── 缓冲区参数（buffer.*）──
    ("buffer.block_time", "block_time_slider", FLOAT, 0.25),
    ("buffer.crossfade_time", "crossfade_slider", FLOAT, 0.05),
    ("buffer.extra_time", "extra_time_slider", FLOAT, 2.5),
    # ── 音频设备参数（audio.*）──
    ("audio.sr_mode", "sr_model_radio", RADIO_SR, "model"),
    ("audio.hostapi", "hostapi_combo", COMBO, ""),
    ("audio.input_device", "input_combo", COMBO, ""),
    ("audio.output_device", "output_combo", COMBO, ""),
    ("audio.output2_device", "output2_combo", COMBO, ""),
    # ── 顶层 ──
    ("model_path", "model_path", ATTR, ""),
    ("hubert", "hubert_combo", COMBO, "chinese"),
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


def params_from_dict(data: dict) -> InferenceParams:
    """从分组 dict 构造 InferenceParams。

    格式：{"voice": {...}, "f0": {...}, "buffer": {...}, "audio": {...},
           "rms_mix": ..., "model_path": ..., "hubert": ...}
    """
    params = InferenceParams()
    voice = data.get("voice", {})
    f0 = data.get("f0", {})
    buf = data.get("buffer", {})
    aud = data.get("audio", {})

    params.voice.formant = voice.get("formant", 0.0)
    params.voice.pitch_map_src_min = voice.get("pitch_map_src_min", 100.0)
    params.voice.pitch_map_src_max = voice.get("pitch_map_src_max", 500.0)
    params.voice.pitch_map_dst_min = voice.get("pitch_map_dst_min", 200.0)
    params.voice.pitch_map_dst_max = voice.get("pitch_map_dst_max", 800.0)

    params.f0.method = f0.get("method", "rmvpe")
    params.f0.rmvpe_threshold = f0.get("rmvpe_threshold", 0.05)
    params.f0.fcpe_confidence_threshold = f0.get("fcpe_confidence_threshold", 0.05)

    params.buffer.block_time = buf.get("block_time", 0.25)
    params.buffer.crossfade_time = buf.get("crossfade_time", 0.05)
    params.buffer.extra_time = buf.get("extra_time", 2.5)

    params.audio.sr_mode = aud.get("sr_mode", "model")
    params.audio.hostapi = aud.get("hostapi", "")
    params.audio.input_device = aud.get("input_device", "")
    params.audio.output_device = aud.get("output_device", "")
    params.audio.output2_device = aud.get("output2_device", "")
    params.audio.enable_out2 = bool(params.audio.output2_device) and params.audio.output2_device != "不使用"

    params.rms_mix = data.get("rms_mix", 0.0)
    params.model_path = data.get("model_path", "")
    params.hubert = data.get("hubert", "chinese")
    return params


def params_to_dict(params: InferenceParams) -> dict:
    """把 InferenceParams 序列化为分组 dict（用于持久化保存）。"""
    return {
        "voice": {
            "formant": params.voice.formant,
            "pitch_map_src_min": params.voice.pitch_map_src_min,
            "pitch_map_src_max": params.voice.pitch_map_src_max,
            "pitch_map_dst_min": params.voice.pitch_map_dst_min,
            "pitch_map_dst_max": params.voice.pitch_map_dst_max,
        },
        "f0": {
            "method": params.f0.method,
            "rmvpe_threshold": params.f0.rmvpe_threshold,
            "fcpe_confidence_threshold": params.f0.fcpe_confidence_threshold,
        },
        "buffer": {
            "block_time": params.buffer.block_time,
            "crossfade_time": params.buffer.crossfade_time,
            "extra_time": params.buffer.extra_time,
        },
        "audio": {
            "sr_mode": params.audio.sr_mode,
            "hostapi": params.audio.hostapi,
            "input_device": params.audio.input_device,
            "output_device": params.audio.output_device,
            "output2_device": params.audio.output2_device,
            "enable_out2": params.audio.enable_out2,
        },
        "rms_mix": params.rms_mix,
        "model_path": params.model_path,
        "hubert": params.hubert,
    }


def _set(win, widget, kind, value):
    """把值写到控件。"""
    if kind == CHECK:
        getattr(win, widget).setChecked(bool(value))
    elif kind == FLOAT:
        slider = getattr(win, widget)
        slider.setValue(float(value))
        # 手动刷新标签：控件未显示时 setValue 可能不触发 valueChanged
        if hasattr(slider, "_update_label"):
            slider._update_label()
    elif kind == INT:
        getattr(win, widget).setValue(int(value))
    elif kind == COMBO:
        idx = getattr(win, widget).findText(str(value))
        if idx >= 0:
            getattr(win, widget).setCurrentIndex(idx)
    elif kind == TEXT:
        getattr(win, widget).setText(str(value))
    elif kind == ATTR:
        setattr(win, widget, str(value) if value else "")
    elif kind == RADIO_F0:
        win.f0_rmvp_btn.setChecked(value == "rmvpe")
        win.f0_fcpe_btn.setChecked(value != "rmvpe")
    elif kind == RADIO_SR:
        win.sr_model_radio.setChecked(value == "model")
        win.sr_device_radio.setChecked(value != "model")
    else:
        raise ValueError(f"未知读写方式: {kind}")


def apply_params(win, params: InferenceParams) -> None:
    """把 InferenceParams 写到所有控件（启动时加载配置调用）。"""
    for path, widget, kind, _default in BINDINGS:
        if widget and hasattr(win, widget):
            value = _get_nested(params, path)
            _set(win, widget, kind, value)

    # 实验参数：F0 阈值滑动条
    if hasattr(win, "exp_f0_threshold_slider"):
        is_rmvpe = params.f0.method == "rmvpe"
        val = params.f0.rmvpe_threshold if is_rmvpe else params.f0.fcpe_confidence_threshold
        slider = win.exp_f0_threshold_slider
        slider.setValue(val)
        if hasattr(slider, "_update_label"):
            slider._update_label()
        if hasattr(win, "exp_f0_threshold_name"):
            win.exp_f0_threshold_name.setText("RMVPE 阈值" if is_rmvpe else "FCPE 阈值")

    # 音域映射 RangeSlider
    if hasattr(win, "exp_pitch_map_src_range"):
        win.exp_pitch_map_src_range.setRange(
            float(params.voice.pitch_map_src_min), float(params.voice.pitch_map_src_max)
        )
    if hasattr(win, "exp_pitch_map_dst_range"):
        win.exp_pitch_map_dst_range.setRange(
            float(params.voice.pitch_map_dst_min), float(params.voice.pitch_map_dst_max)
        )


def collect_params(win) -> InferenceParams:
    """从控件收集 InferenceParams（保存配置时调用）。"""
    params = InferenceParams()
    for path, widget, kind, _default in BINDINGS:
        if not widget or not hasattr(win, widget):
            continue
        w = getattr(win, widget)
        if kind == CHECK:
            value = w.isChecked()
        elif kind in (FLOAT, INT):
            value = w.value()
        elif kind == COMBO:
            value = w.currentText()
        elif kind == TEXT:
            value = w.text()
        elif kind == ATTR:
            value = getattr(win, widget, "")
        elif kind == RADIO_F0:
            value = "rmvpe" if win.f0_rmvp_btn.isChecked() else "fcpe"
        elif kind == RADIO_SR:
            value = "model" if win.sr_model_radio.isChecked() else "device"
        else:
            continue
        _set_nested(params, path, value)

    # 实验参数
    if hasattr(win, "exp_f0_threshold_slider"):
        is_rmvpe = params.f0.method == "rmvpe"
        val = float(win.exp_f0_threshold_slider.value())
        if is_rmvpe:
            params.f0.rmvpe_threshold = val
        else:
            params.f0.fcpe_confidence_threshold = val

    if hasattr(win, "exp_pitch_map_src_range"):
        params.voice.pitch_map_src_min = float(win.exp_pitch_map_src_range.low())
        params.voice.pitch_map_src_max = float(win.exp_pitch_map_src_range.high())
    if hasattr(win, "exp_pitch_map_dst_range"):
        params.voice.pitch_map_dst_min = float(win.exp_pitch_map_dst_range.low())
        params.voice.pitch_map_dst_max = float(win.exp_pitch_map_dst_range.high())

    # enable_out2 是 output2_device 的派生属性，根据当前选择动态计算
    params.audio.enable_out2 = bool(params.audio.output2_device) and params.audio.output2_device != "不使用"

    return params
