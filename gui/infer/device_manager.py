"""设备管理器 — 负责音频设备的枚举和选择"""
from typing import TYPE_CHECKING, List, Tuple
from rvc.audio import get_audio_devices

if TYPE_CHECKING:
    from gui.infer.window import MainWindow


class DeviceManager:
    """管理音频设备的加载和选择"""

    def __init__(self, window: 'MainWindow'):
        self.window = window

    def reload_devices(self) -> None:
        """重新加载音频设备列表"""
        self._refresh_devices()

    def on_hostapi_changed(self, name: str) -> None:
        """音频驱动改变时的处理"""
        if name:
            self._refresh_devices()

    def _refresh_devices(self) -> None:
        """内部：刷新设备列表"""
        hostapi_name = self.window.hostapi_combo.currentText()
        ha_names, ins, outs, in_idx, out_idx = get_audio_devices(hostapi_name)
        self._populate_device_combos(ins, outs)

    def load_hostapis(self) -> None:
        """加载可用的音频驱动"""
        ha_names, ins, outs, in_idx, out_idx = get_audio_devices()
        self.window.hostapi_combo.clear()
        self.window.hostapi_combo.addItems(ha_names)
        self._populate_device_combos(ins, outs)

    def _populate_device_combos(self, ins: List[str], outs: List[str]) -> None:
        """填充设备下拉框"""
        self.window.input_combo.clear()
        self.window.input_combo.addItems(ins)
        self.window.output_combo.clear()
        self.window.output_combo.addItems(outs)
        self.window.output2_combo.clear()
        self.window.output2_combo.addItem("不使用")
        self.window.output2_combo.addItems(outs)
