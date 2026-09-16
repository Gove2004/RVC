"""运行时遥测快照 — GUI 只读取数通道（D5）。

View 层禁止穿透 engine._runner.pipeline.state 四层私有链；
音高/延迟显示统一经 InferController.snapshot() 取本快照对象。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeTelemetry:
    """引擎运行状态的只读瞬时快照。"""
    running: bool
    measure_ms: float   # 端到端实测延迟（毫秒，未运行为 0）
    input_pitch: float  # 当前输入音高（Hz，音域映射前；无声为 0）
