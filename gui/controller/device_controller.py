"""设备目录 — 音频设备的枚举与查询（下拉框位置 ↔ PortAudio 全局索引映射）。"""
import logging
import threading
from typing import List

from gui.state.bindings import OUTPUT2_NONE_TEXT

logger = logging.getLogger(__name__)

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
        from rvc.io.devices import rescan_audio_devices

        try:
            rescan_audio_devices()
        except Exception:
            # 重扫失败不应打断刷新流程，但必须留痕——否则「设备热插拔后刷新没反应」
            # 会完全无线索（PortAudio 内部错误信息只在异常里）
            logger.warning("PortAudio 设备重扫失败，继续按现有缓存刷新列表", exc_info=True)
        self._refresh_devices()

    def on_hostapi_changed(self, name: str) -> None:
        """音频驱动改变时的处理"""
        if name:
            self._refresh_devices()

    def _refresh_devices(self) -> None:
        """内部：刷新所有设备相关下拉框，并尽量保留当前选择。"""
        from rvc.io.devices import get_audio_devices

        ha_names, ins, outs, in_idx, out_idx = get_audio_devices(
            self.window.hostapi_combo.currentText()
        )
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

    def load_hostapis_async(self) -> None:
        """后台线程枚举音频驱动与设备，完成后经 window.devices_ready 信号回主线程填充。

        PortAudio DLL 加载 + 设备枚举 ~170ms（sounddevice import 占大头），
        移出主线程让窗口更早出现（实测本机 43 设备）。rvc.io.devices 的
        import 也一并留在工作线程内完成。
        """
        def _work():
            try:
                from rvc.io.devices import get_audio_devices

                result = get_audio_devices()
            except Exception:
                logger.warning("后台枚举音频设备失败", exc_info=True)
                result = None
            # Qt 信号跨线程自动队列投递，回调在主线程执行
            self.window.devices_ready.emit(result)

        threading.Thread(target=_work, daemon=True, name="device-enum").start()

    def apply_hostapis(self, result) -> None:
        """主线程：填充设备下拉框（devices_ready 信号回调），并恢复配置中的设备选择。

        异步加载时 _load_gui_config 已先跑过，当时下拉框为空、COMBO 回填落空，
        这里从 runtime_params 补一次设备选择恢复。
        关键：填充前必须先把存档值快照到局部——填充空下拉框会触发
        currentTextChanged → 绑定层把「首项」写回 runtime_params，存档值
        会在恢复之前被冲掉（表现为存档配置读不回来，全部回落首项）。
        恢复用 _select 走信号回写，绑定层最终与下拉框一致。
        """
        if result is None:
            return
        ha_names, ins, outs, in_idx, out_idx = result
        self._input_indices = list(in_idx)
        self._output_indices = list(out_idx)
        audio = self.window.runtime_params.audio
        # ── 存档值快照（绑定层会在填充时冲掉 runtime_params 里的这些值）──
        saved_ha = audio.hostapi
        saved_in = audio.input_device
        saved_out = audio.output_device
        saved_out2 = audio.output2_device
        combo = self.window.hostapi_combo
        # 重填 hostapi 列表前屏蔽信号，避免触发 _ha_changed 递归刷新
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(ha_names)
        combo.blockSignals(False)
        if saved_ha and combo.findText(saved_ha) >= 0 and combo.currentText() != saved_ha:
            # 保存的驱动有效且与默认首项不同：放信号恢复选中，触发 _ha_changed
            # 按该驱动重列设备
            combo.setCurrentText(saved_ha)
        else:
            # ① 无保存/已失效（首次启动或设备拔了）；② 保存的驱动就是默认首项
            # （setCurrentText 同值不发信号）。两种情况后台枚举结果本就是当前
            # 驱动过滤的，直接填充设备列表即可；驱动回落事实值保持与下拉框一致
            self._populate_device_combos(ins, outs)
            audio.hostapi = combo.currentText()
        # 恢复快照的设备选择（_select 内 setCurrentIndex 触发绑定信号回写）
        self._select(self.window.input_combo, saved_in)
        self._select(self.window.output_combo, saved_out)
        self._select(self.window.output2_combo, saved_out2)

    def _populate_device_combos(self, ins: List[str], outs: List[str]) -> None:
        """填充设备下拉框"""
        self.window.input_combo.clear()
        self.window.input_combo.addItems(ins)
        self.window.output_combo.clear()
        self.window.output_combo.addItems(outs)
        self.window.output2_combo.clear()
        self.window.output2_combo.addItem(OUTPUT2_NONE_TEXT)  # 占位首项，单源于 bindings
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
