"""离线推理配置数据类"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class OfflineConfig:
    """离线推理配置参数"""
    input_path: str
    output_path: str
    model_path: str
    index_path: str
    pitch: int
    f0method: str
    index_rate: float
    rms_mix: float
    protect: float
    eq_enabled: bool
    eq_bands: Dict[str, float]
    reverb_mix: float

    @classmethod
    def from_ui(cls, window, card) -> 'OfflineConfig':
        """从 UI 组件构造配置对象

        Args:
            window: MainWindow 实例
            card: ModelCard 实例

        Returns:
            OfflineConfig: 配置对象
        """
        from gui.infer.widgets import _sl_value_as_float

        return cls(
            input_path=window.offline_input.text().strip(),
            output_path=window.offline_output.text().strip(),
            model_path=card.pth_edit.text().strip(),
            index_path=card.idx_edit.text().strip(),
            pitch=card.pitch_slider.value(),
            f0method=window.f0_combo.currentText(),
            index_rate=_sl_value_as_float(card.index_rate_slider),
            rms_mix=_sl_value_as_float(card.rms_mix_slider),
            protect=_sl_value_as_float(card.protect_slider),
            eq_enabled=window.eq_enable_checkbox.isChecked(),
            eq_bands={
                'sub': _sl_value_as_float(window.eq_sub_slider),
                'low': _sl_value_as_float(window.eq_low_slider),
                'mid': _sl_value_as_float(window.eq_mid_slider),
                'hi_mid': _sl_value_as_float(window.eq_hi_mid_slider),
                'high': _sl_value_as_float(window.eq_high_slider),
            },
            reverb_mix=_sl_value_as_float(window.reverb_slider),
        )
