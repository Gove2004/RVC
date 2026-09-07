"""GUI 控件 ↔ AppConfig 字段绑定（嵌套路径版）。

把「控件读值 → 配置对象」和「配置对象 → 控件写值」的重复搬运集中到这里，
由 BINDINGS 表驱动，window.py 的 collect/apply 只需一行委托。

路径用点号分隔（如 "inference.pitch"、"engine.block_time"），
支持 AppConfig 的任意嵌套层级。

新增参数时改动点：
1. rvc/config.py 对应 dataclass 加字段
2. BINDINGS 加一行（路径 + 控件 + 读写方式 + 默认值）
3. Tab 里建控件
"""
from rvc.core.config import AppConfig, EngineConfig, InferenceConfig

# 读写方式
CHECK = "check"      # QCheckBox / bool
X100 = "x100"        # QSlider，值 /100
INT = "int"          # QSlider，整数值（不除 100）
COMBO = "combo"      # QComboBox currentText / findText
TEXT = "text"        # QLineEdit text
RADIO_F0 = "radio_f0"  # RMVPE/FCPE 互斥
RADIO_SR = "radio_sr"  # 模型/设备采样率互斥

# 状态字段 schema：(点号路径, window 控件属性名, 读写方式, 缺省默认值)
BINDINGS = [
    # ── 推理参数（inference.*）──
    # 注意：pitch/formant 是模型卡片级参数（card.pitch_slider/card.gender_slider），
    # 不在全局 BINDINGS 表中，由 model_manager 管理持久化。
    ("inference.protect", "protect_slider", X100, 0.5),
    ("inference.f0_method", "f0_rmvp_btn", RADIO_F0, "rmvpe"),
    ("inference.rms_mix", "rms_mix_slider", X100, 0.0),
    ("inference.denoise.enable", "nr_enable_checkbox", CHECK, False),
    ("inference.denoise.strength", "nr_strength_slider", X100, 0.5),
    ("inference.break_protect.enable", "break_enable_checkbox", CHECK, True),
    ("inference.break_protect.src_hz", "break_src_hz_slider", X100, 300.0),
    # ── 引擎参数（engine.*）──
    ("engine.block_time", "block_time_slider", X100, 0.25),
    ("engine.crossfade_time", "crossfade_slider", X100, 0.05),
    ("engine.extra_time", "extra_time_slider", X100, 2.5),
    ("engine.sr_mode", "sr_model_radio", RADIO_SR, "model"),
    ("engine.hostapi", "hostapi_combo", COMBO, ""),
    ("engine.input_device", "input_combo", COMBO, ""),
    ("engine.output_device", "output_combo", COMBO, ""),
    ("engine.output2_device", "output2_combo", COMBO, ""),
    # ── 顶层 ──
    ("active_model", "", TEXT, ""),
]

# 需要按控件步长量化的字段：字段路径 → 量化步长
QUANTIZE = {"inference.break_protect.src_hz": 5.0}


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


def _has_nested(obj, path: str) -> bool:
    """检查 dict 嵌套路径是否存在。"""
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return False
        obj = obj[part]
    return True


def _get_nested_dict(obj, path: str):
    """从 dict 按点号路径递归读取。"""
    for part in path.split("."):
        obj = obj[part]
    return obj


def _set_nested_dict(obj, path: str, value):
    """向 dict 按点号路径递归写入（自动创建中间 dict）。"""
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in obj:
            obj[part] = {}
        obj = obj[part]
    obj[parts[-1]] = value


# 旧短键格式 → 新嵌套路径的映射（用于一次性迁移）
_OLD_KEY_MAPPING = {
    "protect": "inference.protect",
    "f0": "inference.f0_method",
    "rms": "inference.rms_mix",
    "nr_en": "inference.denoise.enable",
    "nr_str": "inference.denoise.strength",
    "brk_en": "inference.break_protect.enable",
    "brk_hz": "inference.break_protect.src_hz",
    "bl": "engine.block_time",
    "cf": "engine.crossfade_time",
    "ex": "engine.extra_time",
    "sr_mode": "engine.sr_mode",
    "ha": "engine.hostapi",
    "in_dev": "engine.input_device",
    "out_dev": "engine.output_device",
    "out2_dev": "engine.output2_device",
    "active_model": "active_model",
}


def _migrate_old_format(data: dict) -> dict:
    """检测旧短键格式，自动转换成嵌套结构。已经是新格式则原样返回。

    判断标准：同时存在多个旧短键（如 nr_en + bl）才认为是旧格式，
    避免新格式里的 active_model 单独触发误迁移。
    """
    old_keys_present = [k for k in _OLD_KEY_MAPPING if k in data]
    # 旧格式至少有 3 个以上短键（新格式只有 active_model 一个顶层键可能重合）
    if len(old_keys_present) < 3:
        return data
    result = dict(data)  # 先复制所有键，避免丢失非配置字段
    for old_key, new_path in _OLD_KEY_MAPPING.items():
        if old_key in result:
            _set_nested_dict(result, new_path, result.pop(old_key))
    return result


def _parse(kind, raw):
    if kind == CHECK:
        return bool(raw)
    if kind in (X100, INT):
        return float(raw) if kind == X100 else int(raw)
    return str(raw)


def state_from_dict(data: dict) -> AppConfig:
    """从持久化字典（嵌套结构）构造 AppConfig（并按控件步长量化）。

    自动检测旧短键格式并迁移到嵌套结构。
    """
    data = _migrate_old_format(data)
    cfg = AppConfig()
    for path, _w, kind, default in BINDINGS:
        raw = _get_nested_dict(data, path) if _has_nested(data, path) else default
        val = _parse(kind, raw)
        step = QUANTIZE.get(path)
        if step:
            val = round(val / step) * step
        _set_nested(cfg, path, val)
    # enable_out2 不在 BINDINGS 表（无独立控件），根据是否选择了副输出设备自动推导
    cfg.engine.enable_out2 = bool(cfg.engine.output2_device) and cfg.engine.output2_device != "不使用"
    return cfg


def state_to_dict(state: AppConfig) -> dict:
    """AppConfig → 持久化字典（嵌套结构）。"""
    result = {}
    for path, _w, _k, _d in BINDINGS:
        _set_nested_dict(result, path, _get_nested(state, path))
    return result


def _get(win, widget, kind):
    w = getattr(win, widget)
    if kind == CHECK:
        return w.isChecked()
    if kind == X100:
        return w.value() / 100.0
    if kind == INT:
        return w.value()
    if kind == COMBO:
        return w.currentText()
    if kind == TEXT:
        return w.text().strip()
    if kind == RADIO_F0:
        return "rmvpe" if w.isChecked() else "fcpe"
    if kind == RADIO_SR:
        return "model" if w.isChecked() else "device"
    raise ValueError(f"未知读写方式: {kind}")


def _set(win, widget, kind, value):
    if kind == CHECK:
        getattr(win, widget).setChecked(bool(value))
    elif kind == X100:
        getattr(win, widget).setValue(int(round(float(value) * 100)))
    elif kind == INT:
        getattr(win, widget).setValue(int(value))
    elif kind == COMBO:
        idx = getattr(win, widget).findText(str(value))
        if idx >= 0:
            getattr(win, widget).setCurrentIndex(idx)
    elif kind == TEXT:
        getattr(win, widget).setText(str(value))
    elif kind == RADIO_F0:
        win.f0_rmvp_btn.setChecked(value == "rmvpe")
        win.f0_fcpe_btn.setChecked(value != "rmvpe")
    elif kind == RADIO_SR:
        win.sr_model_radio.setChecked(value == "model")
        win.sr_device_radio.setChecked(value != "model")
    else:
        raise ValueError(f"未知读写方式: {kind}")


def collect_gui_state(win) -> AppConfig:
    """从控件收集完整配置（含 active_model 特例）。"""
    cfg = AppConfig()
    for path, widget, kind, _d in BINDINGS:
        if widget:
            _set_nested(cfg, path, _get(win, widget, kind))
    # enable_out2 无独立控件，根据 output2_combo 是否选了"不使用"自动推导
    out2_text = win.output2_combo.currentText()
    cfg.engine.enable_out2 = bool(out2_text) and out2_text != "不使用"
    active = ""
    card = win.model_manager.active_card
    if card is not None:
        active = card.pth_edit.text().strip()
    cfg.active_model = active
    return cfg


def apply_gui_state(win, state: AppConfig) -> None:
    """将配置写回控件（含 active_model 特例）。"""
    for path, widget, kind, _d in BINDINGS:
        if widget:
            _set(win, widget, kind, _get_nested(state, path))
    if state.active_model:
        for card in win.model_manager.cards:
            if card.pth_edit.text().strip() == state.active_model:
                card.set_active(True)
                win.model_manager.active_card = card
                break


def runtime_from_state(state: AppConfig) -> InferenceConfig:
    """从 AppConfig 提取推理参数（engine 消费的配置）。

    注意：副输出 enable_out2 属于 EngineConfig，推理时由 engine 直接读取，
    此处只返回 InferenceConfig。
    """
    return state.inference


def engine_from_state(state: AppConfig) -> EngineConfig:
    """从 AppConfig 提取引擎参数。"""
    return state.engine


def gender_to_formant(v: float) -> float:
    """性别滑杆值 [0,1] → formant shift [-2.5, +2.5]。唯一换算来源。"""
    return (v - 0.5) * 5


def formant_to_gender(f: float) -> float:
    """gender_to_formant 的反函数。"""
    return f / 5.0 + 0.5


def format_error_message(error: Exception | str) -> str:
    """格式化错误消息，只保留最后一行有意义的内容。"""
    msg = str(error).strip()
    lines = msg.splitlines()
    return lines[-1] if lines else "未知错误"
