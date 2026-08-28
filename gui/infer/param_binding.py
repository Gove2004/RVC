"""GUI 控件 ↔ InferGuiState 字段绑定。

把「控件读值 → 状态对象」和「状态对象 → 控件写值」的重复搬运集中到这里，
由 BINDINGS 表驱动，window.py 的 collect/apply 只需一行委托。

新增参数时改动点（替代原先 7 处手写）：
1. InferGuiState 加字段（gui/configs/infer_state.py）
2. BINDINGS 加一行（本文件）
3. from_dict/to_dict 短键（gui/configs/infer_state.py）—— 存储格式层
4. Tab 里建控件

f0method / sr_mode 是互斥 radio，active_model 来自模型卡片，均走特例。
"""
from gui.configs.infer_state import InferGuiState
from gui.infer.controller import RuntimeConfig

# 读写方式
CHECK = "check"      # QCheckBox / bool
X100 = "x100"        # QSlider，值 /100
COMBO = "combo"      # QComboBox currentText / findText
TEXT = "text"        # QLineEdit text
RADIO_F0 = "radio_f0"  # RMVPE/FCPE 互斥
RADIO_SR = "radio_sr"  # 模型/设备采样率互斥

# 状态字段 schema：一条记录同时驱动 控件同步（collect/apply）、持久化（from/to_dict）、
# 类型转换（kind → bool/float/str）。
# (InferGuiState 字段名, window 控件属性名, 读写方式, 存储短键, 缺省默认值)
BINDINGS = [
    ("block_time", "block_time_slider", X100, "bl", 0.25),
    ("crossfade_time", "crossfade_slider", X100, "cf", 0.05),
    ("extra_time", "extra_time_slider", X100, "ex", 2.5),
    ("protect", "protect_slider", X100, "protect", 0.25),
    ("f0method", "f0_rmvp_btn", RADIO_F0, "f0", "fcpe"),
    ("sr_mode", "sr_model_radio", RADIO_SR, "sr_mode", "model"),
    ("rms_mix", "rms_mix_slider", X100, "rms", 0.0),
    ("nr_enable", "nr_enable_checkbox", CHECK, "nr_en", False),
    ("nr_strength", "nr_strength_slider", X100, "nr_str", 0.5),
    ("break_enable", "break_enable_checkbox", CHECK, "brk_en", True),
    ("break_src_hz", "break_src_hz_slider", X100, "brk_hz", 300.0),
    ("hostapi", "hostapi_combo", COMBO, "ha", ""),
    ("input_device", "input_combo", COMBO, "in_dev", ""),
    ("output_device", "output_combo", COMBO, "out_dev", ""),
    ("output2_device", "output2_combo", COMBO, "out2_dev", ""),
    ("active_model", "", TEXT, "active_model", ""),
]


def _parse(kind, raw):
    """按读写方式把存储值转回状态字段类型"""
    if kind == CHECK:
        return bool(raw)
    if kind == X100:
        return float(raw)
    return str(raw)


def state_from_dict(data: dict) -> InferGuiState:
    """持久化字典（短键）→ 状态对象"""
    return InferGuiState(**{field: _parse(kind, data.get(key, default))
                            for field, _w, kind, key, default in BINDINGS})


def state_to_dict(state: InferGuiState) -> dict:
    """状态对象 → 持久化字典（短键）"""
    d = {key: getattr(state, field)
         for field, _w, _k, key, _d in BINDINGS}
    return d


def _get(win, widget, kind):
    w = getattr(win, widget)
    if kind == CHECK:
        return w.isChecked()
    if kind == X100:
        return w.value() / 100.0
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
        getattr(win, widget).setValue(int(round(value * 100)))
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


def collect_gui_state(win) -> InferGuiState:
    """从控件收集完整 GUI 状态（含 active_model 特例）"""
    kw = {field: _get(win, widget, kind) for field, widget, kind, _k, _d in BINDINGS
          if widget}
    active = ""
    card = win.model_manager.active_card
    if card is not None:
        active = card.pth_edit.text().strip()
    kw["active_model"] = active
    return InferGuiState(**kw)


def apply_gui_state(win, state: InferGuiState) -> None:
    """将状态写回控件（含 active_model 特例）"""
    for field, widget, kind, _k, _d in BINDINGS:
        if widget:
            _set(win, widget, kind, getattr(state, field))
    if state.active_model:
        for card in win.model_manager.cards:
            if card.pth_edit.text().strip() == state.active_model:
                card.set_active(True)
                win.model_manager.active_card = card
                break


def runtime_from_state(state: InferGuiState) -> RuntimeConfig:
    """状态对象 → 运行时参数（engine 消费的子集）"""
    return RuntimeConfig(
        # output2_combo 首项为「不使用」；combo 为空（无设备）时旧逻辑 index=-1 视为未启用
        enable_out2=state.output2_device not in ("", "不使用"),
        rms_mix=state.rms_mix,
        nr_enable=state.nr_enable,
        nr_strength=state.nr_strength,
        break_enable=state.break_enable,
        break_src_hz=state.break_src_hz,
    )
