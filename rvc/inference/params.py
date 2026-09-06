"""运行时参数容器 — 兼容层。

新配置体系在 rvc/config.py（InferenceConfig 嵌套结构）。
本模块提供旧字段名（gender/f0method/nr_enable/break_enable 等）到新字段的
属性映射，供过渡阶段使用。最终目标是全部引用直接用 InferenceConfig，
删除本兼容层。
"""
from rvc.config import InferenceConfig

HUBERT_DEFAULT = "chinese"


class Params(InferenceConfig):
    """旧字段名兼容层 — 扁平字段映射到 InferenceConfig 的嵌套字段。"""

    # gender → formant
    @property
    def gender(self):
        return self.formant

    @gender.setter
    def gender(self, v):
        self.formant = v

    # f0method → f0_method
    @property
    def f0method(self):
        return self.f0_method

    @f0method.setter
    def f0method(self, v):
        self.f0_method = v

    # nr_enable → denoise.enable
    @property
    def nr_enable(self):
        return self.denoise.enable

    @nr_enable.setter
    def nr_enable(self, v):
        self.denoise.enable = v

    # nr_strength → denoise.strength
    @property
    def nr_strength(self):
        return self.denoise.strength

    @nr_strength.setter
    def nr_strength(self, v):
        self.denoise.strength = v

    # break_enable → break_protect.enable
    @property
    def break_enable(self):
        return self.break_protect.enable

    @break_enable.setter
    def break_enable(self, v):
        self.break_protect.enable = v

    # break_src_hz → break_protect.src_hz
    @property
    def break_src_hz(self):
        return self.break_protect.src_hz

    @break_src_hz.setter
    def break_src_hz(self, v):
        self.break_protect.src_hz = v
