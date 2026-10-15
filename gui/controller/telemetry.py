"""运行时遥测快照 — GUI 只读取数通道。

View 层禁止穿透 engine._runner.pipeline.state 四层私有链；
音高/延迟显示统一经 InferController.snapshot() 取本快照对象。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeTelemetry:
    """引擎运行状态的只读瞬时快照。"""
    running: bool
    measure_ms: float   # 端到端延迟（毫秒，硬件时间戳+推理补偿，未运行为 0）
    input_pitch: float  # 当前输入音高（Hz，音域映射前；无声为 0）
    error_count: int = 0  # 实时链路音频回调累计错误数（0 = 无错误）
    driver_in_ms: float = 0.0  # 输入驱动延迟（ADC→回调，含块缓冲）
    driver_out_ms: float = 0.0  # 输出驱动延迟（回调→DAC，含系统混音器）
    infer_ms: float = 0.0  # 推理耗时（GPU 端到端）
