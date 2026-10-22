"""设备目录 — 音频设备的枚举与查询（下拉框位置 ↔ PortAudio 全局索引映射）。

分层契约：本类是 controller，唯一 UI 依赖是 `gui.view.contracts.DeviceViewHost`
（MainWindow 实现）。本类提供枚举数据与「保留选择」的决策；下拉框重填、
信号屏蔽、占位项等控件机械操作在 view 层的 device_populate 完成。
"""
import logging
from typing import TYPE_CHECKING, List

from rvc.io.devices import get_audio_devices, rescan_audio_devices

if TYPE_CHECKING:
    from gui.view.contracts import DeviceViewHost

logger = logging.getLogger(__name__)

class DeviceCatalog:
    """枚举音频设备并维护「下拉框位置 → PortAudio 索引」映射。"""

    def __init__(self, host: "DeviceViewHost"):
        self.host = host
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
            # 重扫失败不应打断刷新流程，但必须留痕——否则「设备热插拔后刷新没反应」
            # 会完全无线索（PortAudio 内部错误信息只在异常里）
            logger.warning("PortAudio 设备重扫失败，继续按现有缓存刷新列表", exc_info=True)
        self._refresh_devices()

    def on_hostapi_changed(self, name: str) -> None:
        """音频驱动改变时的处理"""
        if name:
            self._refresh_devices()

    def load_hostapis(self) -> None:
        """加载可用的音频驱动（同步：PortAudio 加载 ~170ms，在窗口出现前完成）。

        调用时下拉框全为空，「保留当前选择」退化为首项默认；hostapi 重填
        期间的信号屏蔽在 device_populate 内部处理（启动时不触发递归刷新）。
        """
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        """按当前 hostapi 过滤枚举设备，交宿主重填并恢复选择。"""
        prev = self.host.device_current_selections()
        ha_names, ins, outs, in_idx, out_idx = get_audio_devices(prev[0])
        self._input_indices = list(in_idx)
        self._output_indices = list(out_idx)
        self.host.device_populate(ha_names, ins, outs, prev)

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
