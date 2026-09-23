"""设备目录 — 音频设备的枚举与查询（下拉框位置 ↔ PortAudio 全局索引映射）。"""
from typing import List
from rvc.io.devices import get_audio_devices, rescan_audio_devices

class DeviceCatalog:
    """枚举音频设备并维护「下拉框位置 → PortAudio 索引」映射。"""

    def __init__(self, window: 'MainWindow'):
        self.window = window
        # PortAudio 全局设备索引映射（下拉框位置 → PortAudio 索引）
        # get_audio_devices 返回的 in_idx/out_idx 是 PortAudio 全局索引，
        # 下拉框 currentIndex() 只是过滤后的位置，必须通过映射转换。
        self._input_indices: List[int] = []
        self._output_indices: List[int] = []

    def reload_devices(self) -> None:
        """刷新按钮：强制 PortAudio 重新枚举（支持设备热插拔），再刷新列表。"""
        # PortAudio 会缓存设备列表；设备热插拔后必须强制重扫（唯一入口见 rvc/io/devices），
        # 否则 query_devices() 始终返回旧列表，刷新看起来「没反应」。
        try:
            rescan_audio_devices()
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
        self._input_indices = list(in_idx)
        self._output_indices = list(out_idx)
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
        self._input_indices = list(in_idx)
        self._output_indices = list(out_idx)
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

    def get_input_device_index(self, combo_idx: int) -> int:
        """下拉框位置 → PortAudio 全局设备索引。

        输入下拉框恒为 0 基位置；越界说明下拉状态与设备表不一致，直接抛错而非静默回退。
        """
        if 0 <= combo_idx < len(self._input_indices):
            return self._input_indices[combo_idx]
        raise IndexError(f"输入设备位置越界: {combo_idx}（共 {len(self._input_indices)} 项）")

    def get_output_device_index(self, combo_idx: int) -> int:
        """下拉框位置 → PortAudio 全局设备索引。

        combo_idx=-1 是副输出「不使用」的有意取值（调用方据此转 None）；
        其余越界说明下拉状态与设备表不一致，直接抛错而非静默回退。
        """
        if combo_idx == -1:
            return -1
        if 0 <= combo_idx < len(self._output_indices):
            return self._output_indices[combo_idx]
        raise IndexError(f"输出设备位置越界: {combo_idx}（共 {len(self._output_indices)} 项）")
