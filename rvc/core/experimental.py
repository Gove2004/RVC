"""实验功能配置 — 运行时可调节的音质优化参数。

所有实验性功能的开关和参数都在这里，GUI 实验 Tab 直接修改此对象的属性，
推理代码每次调用时读取，实现运行时实时调节，无需重启。
"""


class ExperimentalConfig:
    """实验功能配置（运行时可修改，可持久化）。"""

    def __init__(self):
        # ── 辅音保护软阈值（解决咬字不清） ──
        self.protect_soft_enabled = True
        self.protect_soft_threshold_hz = 10.0  # 过渡中心（Hz）
        self.protect_soft_width = 20.0  # 过渡宽度（Hz，sigmoid 的 4σ）
        # ── F0 提取清浊判定阈值 ──
        self.rmvpe_threshold = 0.03  # RMVPE thred，越高越容易判清音（UV）
        self.fcpe_confidence_threshold = 0.025  # FCPE confidence，越高越容易判清音（UV）

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）。"""
        return {
            "protect_soft_enabled": self.protect_soft_enabled,
            "protect_soft_threshold_hz": self.protect_soft_threshold_hz,
            "protect_soft_width": self.protect_soft_width,
            "rmvpe_threshold": self.rmvpe_threshold,
            "fcpe_confidence_threshold": self.fcpe_confidence_threshold,
        }

    def from_dict(self, data: dict) -> None:
        """从字典反序列化（缺失字段保持默认值）。"""
        if "protect_soft_enabled" in data:
            self.protect_soft_enabled = bool(data["protect_soft_enabled"])
        if "protect_soft_threshold_hz" in data:
            self.protect_soft_threshold_hz = float(data["protect_soft_threshold_hz"])
        if "protect_soft_width" in data:
            self.protect_soft_width = float(data["protect_soft_width"])
        if "rmvpe_threshold" in data:
            self.rmvpe_threshold = float(data["rmvpe_threshold"])
        if "fcpe_confidence_threshold" in data:
            self.fcpe_confidence_threshold = float(data["fcpe_confidence_threshold"])


# 全局单例，GUI 和推理代码共用
experimental_config = ExperimentalConfig()
