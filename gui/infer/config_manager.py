"""配置管理器 — 负责加载和保存 GUI 配置"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from gui.configs import load_config, save_config

if TYPE_CHECKING:
    from gui.infer.window import MainWindow

CONFIG_VERSION = 2  # 配置文件版本号


class ConfigManager:
    """管理 GUI 配置的加载和保存"""

    def __init__(self, window: 'MainWindow'):
        self.window = window

    def load_config(self) -> None:
        """从持久化存储加载配置"""
        cfg = load_config()
        d = cfg.get("gui", {})
        self._load_preset_config(d)
        self._load_device_config(d)
        self._load_engine_config(d)
        self._load_audio_config(d)
        self._load_active_model(d)

    def save_config(self) -> None:
        """保存配置到持久化存储"""
        from gui.infer.widgets import _sl_value_as_float

        # 保存当前选中模型的 pth 路径
        active_pth = ""
        if self.window.model_manager.active_card:
            active_pth = self.window.model_manager.active_card.pth_edit.text().strip()

        d = {
            "version": CONFIG_VERSION,
            "bl": _sl_value_as_float(self.window.block_time_slider),
            "cf": _sl_value_as_float(self.window.crossfade_slider),
            "ex": _sl_value_as_float(self.window.extra_time_slider),
            "f0": self.window.f0_combo.currentText(),
            "sr_mode": "model" if self.window.sr_model_radio.isChecked() else "device",
            "eq_en": self.window.eq_enable_checkbox.isChecked(),
            "eq_sub": _sl_value_as_float(self.window.eq_sub_slider),
            "eq_lo": _sl_value_as_float(self.window.eq_low_slider),
            "eq_mi": _sl_value_as_float(self.window.eq_mid_slider),
            "eq_hi_mid": _sl_value_as_float(self.window.eq_hi_mid_slider),
            "eq_hi": _sl_value_as_float(self.window.eq_high_slider),
            "rev": _sl_value_as_float(self.window.reverb_slider),
            "preset": self.window.preset_combo.currentText(),
            "ha": self.window.hostapi_combo.currentText(),
            "in_dev": self.window.input_combo.currentText(),
            "out_dev": self.window.output_combo.currentText(),
            "out2_dev": self.window.output2_combo.currentText(),
            "active_model": active_pth,
        }
        cfg = load_config()
        cfg["gui"] = d
        save_config(cfg)

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
            idx = self.window.hostapi_combo.findText(ha)
            if idx >= 0:
                self.window.hostapi_combo.setCurrentIndex(idx)

        for key, combo in [
            ("in_dev", self.window.input_combo),
            ("out_dev", self.window.output_combo),
            ("out2_dev", self.window.output2_combo),
        ]:
            dev = d.get(key)
            if dev:
                idx = combo.findText(dev)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _load_engine_config(self, d: Dict[str, Any]) -> None:
        """加载引擎参数配置"""
        self.window.block_time_slider.setValue(int(d.get("bl", 0.25) * 100))
        self.window.crossfade_slider.setValue(int(d.get("cf", 0.05) * 100))
        self.window.extra_time_slider.setValue(int(d.get("ex", 2.5) * 100))

        f0 = d.get("f0", "fcpe")
        idx = self.window.f0_combo.findText(f0)
        if idx >= 0:
            self.window.f0_combo.setCurrentIndex(idx)

        # 加载采样率模式
        sr_mode = d.get("sr_mode", "model")
        if sr_mode == "model":
            self.window.sr_model_radio.setChecked(True)
        else:
            self.window.sr_device_radio.setChecked(True)

    def _load_audio_config(self, d: Dict[str, Any]) -> None:
        """加载音频效果配置"""
        self.window.eq_enable_checkbox.setChecked(d.get("eq_en", False))
        self.window.eq_sub_slider.setValue(int(d.get("eq_sub", 0.0) * 100))
        self.window.eq_low_slider.setValue(int(d.get("eq_lo", 0.0) * 100))
        self.window.eq_mid_slider.setValue(int(d.get("eq_mi", 0.0) * 100))
        self.window.eq_hi_mid_slider.setValue(int(d.get("eq_hi_mid", 0.0) * 100))
        self.window.eq_high_slider.setValue(int(d.get("eq_hi", 0.0) * 100))
        self.window.reverb_slider.setValue(int(d.get("rev", 0.0) * 100))

    def _load_active_model(self, d: Dict[str, Any]) -> None:
        """加载上次选中的模型"""
        active_pth = d.get("active_model", "")
        if not active_pth:
            return
        # 在模型列表中查找匹配的模型并设置为 active
        for card in self.window.model_manager.cards:
            if card.pth_edit.text().strip() == active_pth:
                card.set_active(True)
                self.window.model_manager.active_card = card
                break
