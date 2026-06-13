"""配置管理器 — 负责加载和保存 GUI 配置"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from gui.configs import load_state_json, save_state_json

if TYPE_CHECKING:
    from gui.infer.window import MainWindow

CONFIG_KEY = "gui"


class ConfigManager:
    """管理 GUI 配置的加载和保存"""

    def __init__(self, window: 'MainWindow'):
        self.window = window

    def load_config(self) -> None:
        """从持久化存储加载配置"""
        d = load_state_json(CONFIG_KEY, {})
        self._load_preset_config(d)
        self._load_device_config(d)
        self._load_engine_config(d)
        self._load_audio_config(d)

    def save_config(self) -> None:
        """保存配置到持久化存储"""
        from gui.infer.widgets import _sl_value_as_float

        d = {
            "version": 2,
            "bl": _sl_value_as_float(self.window.bl_sl),
            "cf": _sl_value_as_float(self.window.cf_sl),
            "ex": _sl_value_as_float(self.window.ex_sl),
            "f0": self.window.f0_combo.currentText(),
            "eq_en": self.window.eq_en.isChecked(),
            "eq_sub": _sl_value_as_float(self.window.eq_sub),
            "eq_lo": _sl_value_as_float(self.window.eq_lo),
            "eq_mi": _sl_value_as_float(self.window.eq_mi),
            "eq_hi_mid": _sl_value_as_float(self.window.eq_hi_mid),
            "eq_hi": _sl_value_as_float(self.window.eq_hi),
            "rev": _sl_value_as_float(self.window.rev_sl),
            "preset": self.window.preset_combo.currentText(),
            "ha": self.window.ha_combo.currentText(),
            "in_dev": self.window.in_combo.currentText(),
            "out_dev": self.window.out_combo.currentText(),
            "out2_dev": self.window.out2_combo.currentText(),
        }
        save_state_json(CONFIG_KEY, d)

    def _load_preset_config(self, d: Dict[str, Any]) -> None:
        """加载预设配置"""
        preset = d.get("preset", "默认")
        idx = self.window.preset_combo.findText(preset)
        if idx >= 0:
            self.window.preset_combo.setCurrentIndex(idx)

    def _load_device_config(self, d: Dict[str, Any]) -> None:
        """加载音频设备配置"""
        ha = d.get("ha")
        if ha:
            idx = self.window.ha_combo.findText(ha)
            if idx >= 0:
                self.window.ha_combo.setCurrentIndex(idx)

        for key, combo in [
            ("in_dev", self.window.in_combo),
            ("out_dev", self.window.out_combo),
            ("out2_dev", self.window.out2_combo),
        ]:
            dev = d.get(key)
            if dev:
                idx = combo.findText(dev)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _load_engine_config(self, d: Dict[str, Any]) -> None:
        """加载引擎参数配置"""
        self.window.bl_sl.setValue(int(d.get("bl", 0.25) * 100))
        self.window.cf_sl.setValue(int(d.get("cf", 0.05) * 100))
        self.window.ex_sl.setValue(int(d.get("ex", 2.5) * 100))

        f0 = d.get("f0", "fcpe")
        idx = self.window.f0_combo.findText(f0)
        if idx >= 0:
            self.window.f0_combo.setCurrentIndex(idx)

    def _load_audio_config(self, d: Dict[str, Any]) -> None:
        """加载音频效果配置"""
        self.window.eq_en.setChecked(d.get("eq_en", False))
        self.window.eq_sub.setValue(int(d.get("eq_sub", 0) * 100))
        self.window.eq_lo.setValue(int(d.get("eq_lo", 0) * 100))
        self.window.eq_mi.setValue(int(d.get("eq_mi", 0) * 100))
        self.window.eq_hi_mid.setValue(int(d.get("eq_hi_mid", 0) * 100))
        self.window.eq_hi.setValue(int(d.get("eq_hi", 0) * 100))
        self.window.rev_sl.setValue(int(d.get("rev", 0) * 100))
