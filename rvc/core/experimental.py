"""实验功能配置 — 运行时可调节的音质优化参数。

所有实验性功能的参数都在这里，GUI 实验 Tab 直接修改此对象的属性，
推理代码每次调用时读取，实现运行时实时调节，无需重启。

辅音保护（渐变式）和音域映射（半音尺度）始终生效，无需启用开关。
"""


class ExperimentalConfig:
    """实验功能配置（运行时可修改，可持久化）。"""

    def __init__(self):
        # ── 辅音保护（渐变式，始终生效） ──
        self.rmvpe_threshold = 0.05  # RMVPE 清浊判定阈值
        self.fcpe_confidence_threshold = 0.05  # FCPE 清浊判定阈值
        self.protect_soft_threshold_hz = 20.0  # 过渡中心（Hz）
        self.protect_soft_width = 30.0  # 过渡宽度（Hz，sigmoid 的 4σ）
        # ── 音域映射（半音尺度，始终生效，替代固定 pitch 偏移） ──
        # 把原声音域线性映射到目标音域（在 MIDI 半音尺度上，保持音程不变）
        self.pitch_map_src_min = 100.0   # 原声音域下限 (Hz)
        self.pitch_map_src_max = 500.0   # 原声音域上限 (Hz)
        self.pitch_map_dst_min = 200.0   # 目标音域下限 (Hz)
        self.pitch_map_dst_max = 800.0   # 目标音域上限 (Hz)
        # ── 清浊分析（F0 confidence + 中值滤波，始终生效） ──
        # 学术依据：Morrison penn 连续 periodicity、Graf 2015 多特征融合、Takeda ATR 边界检测
        # 中值滤波窗口固定为 3（30ms），无需调节

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）。"""
        return {
            "rmvpe_threshold": self.rmvpe_threshold,
            "fcpe_confidence_threshold": self.fcpe_confidence_threshold,
            "protect_soft_threshold_hz": self.protect_soft_threshold_hz,
            "protect_soft_width": self.protect_soft_width,
            "pitch_map_src_min": self.pitch_map_src_min,
            "pitch_map_src_max": self.pitch_map_src_max,
            "pitch_map_dst_min": self.pitch_map_dst_min,
            "pitch_map_dst_max": self.pitch_map_dst_max,
        }

    def from_dict(self, data: dict) -> None:
        """从字典反序列化（缺失字段保持默认值）。"""
        if "rmvpe_threshold" in data:
            self.rmvpe_threshold = float(data["rmvpe_threshold"])
        if "fcpe_confidence_threshold" in data:
            self.fcpe_confidence_threshold = float(data["fcpe_confidence_threshold"])
        if "protect_soft_threshold_hz" in data:
            self.protect_soft_threshold_hz = float(data["protect_soft_threshold_hz"])
        if "protect_soft_width" in data:
            self.protect_soft_width = float(data["protect_soft_width"])
        if "pitch_map_src_min" in data:
            self.pitch_map_src_min = float(data["pitch_map_src_min"])
        if "pitch_map_src_max" in data:
            self.pitch_map_src_max = float(data["pitch_map_src_max"])
        if "pitch_map_dst_min" in data:
            self.pitch_map_dst_min = float(data["pitch_map_dst_min"])
        if "pitch_map_dst_max" in data:
            self.pitch_map_dst_max = float(data["pitch_map_dst_max"])


# 全局单例，GUI 和推理代码共用
experimental_config = ExperimentalConfig()
