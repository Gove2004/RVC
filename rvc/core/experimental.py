"""实验功能配置 — 运行时可调节的音质优化参数。

所有实验性功能的开关和参数都在这里，GUI 实验 Tab 直接修改此对象的属性，
推理代码每次调用时读取，实现运行时实时调节，无需重启。
"""


class ExperimentalConfig:
    """实验功能配置（运行时可修改）。"""

    def __init__(self):
        # ── 辅音保护软阈值（解决咬字不清） ──
        self.protect_soft_enabled = True
        self.protect_soft_threshold_hz = 10.0  # 过渡中心（Hz）
        self.protect_soft_width = 20.0  # 过渡宽度（Hz，sigmoid 的 4σ）


# 全局单例，GUI 和推理代码共用
experimental_config = ExperimentalConfig()
