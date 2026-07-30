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
        """刷新按钮：强制 PortAudio 重新枚举（支持设备热插拔），再刷新列表。"""
        # PortAudio 会缓存设备列表；设备热插拔后必须用 _terminate + _initialize
        # 强制重扫，否则 query_devices() 始终返回旧列表，刷新看起来「没反应」。
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        self._refresh_devices()

    def on_hostapi_changed(self, name: str) -> None:
        """音频驱动改变时的处理"""
        if name:
            self._refresh_devices()

    def _refresh_devices(self) -> None:
        """内部：刷新所有设备相关下拉框，并尽量保留当前选择。"""
        hostapi_name = self.window.hostapi_combo.currentText()
        ha_names, ins, outs, in_idx, out_idx = get_audio_devices(hostapi_name)
        prev_ha = self.window.hostapi_combo.currentText()
        prev_in = self.window.input_combo.currentText()
        prev_out = self.window.output_combo.currentText()
        prev_out2 = self.window.output2_combo.currentText()
        # 重填 hostapi 列表前屏蔽信号，避免触发 _ha_changed 递归刷新
        self.window.hostapi_combo.blockSignals(True)
        self.window.hostapi_combo.clear()
        self.window.hostapi_combo.addItems(ha_names)
        self._select(self.window.hostapi_combo, prev_ha)
        self.window.hostapi_combo.blockSignals(False)
        self._populate_device_combos(ins, outs)
        self._select(self.window.input_combo, prev_in)
        self._select(self.window.output_combo, prev_out)
        self._select(self.window.output2_combo, prev_out2)

    @staticmethod
    def _select(combo, text):
        """若 text 仍在列表中则恢复选中，否则保持默认首项。"""
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

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
