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
        hostapi_name = self.window.ha_combo.currentText()
        if not hostapi_name:
            return
        apis, ins, outs = get_audio_devices()
        self._populate_device_combos(ins, outs)

    def on_hostapi_changed(self, name: str) -> None:
        """音频驱动改变时的处理"""
        if not name:
            return
        apis, ins, outs = get_audio_devices()
        self._populate_device_combos(ins, outs)

    def load_hostapis(self) -> None:
        """加载可用的音频驱动"""
        apis, ins, outs = get_audio_devices()
        self.window.ha_combo.clear()
        self.window.ha_combo.addItems([a.name for a in apis])
        self._populate_device_combos(ins, outs)

    def _populate_device_combos(self, ins: List, outs: List) -> None:
        """填充设备下拉框"""
        self.window.in_combo.clear()
        self.window.in_combo.addItems([d.name for d in ins])
        self.window.out_combo.clear()
        self.window.out_combo.addItems([d.name for d in outs])
        self.window.out2_combo.clear()
        self.window.out2_combo.addItem("不使用")
        self.window.out2_combo.addItems([d.name for d in outs])
