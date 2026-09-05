"""推理 GUI 状态对象（纯数据容器）。

控件同步（collect/apply）与持久化（from/to_dict）由 gui.infer.param_binding 的
BINDINGS 表统一驱动，新增字段时只需在 BINDINGS 加一行 + 此处加一个字段。
"""
from dataclasses import dataclass


@dataclass
class InferGuiState:
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    protect: float = 0.5
    f0method: str = "rmvpe"
    sr_mode: str = "model"
    rms_mix: float = 0.0
    nr_enable: bool = False
    nr_strength: float = 0.5
    break_enable: bool = True
    break_src_hz: float = 300.0
    hostapi: str = ""
    input_device: str = ""
    output_device: str = ""
    output2_device: str = ""
    active_model: str = ""
