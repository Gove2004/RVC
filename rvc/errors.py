"""RVC 错误类型。"""


class ModelLoadError(Exception):
    """模型加载失败（pth 损坏/缺失/不兼容）。"""
