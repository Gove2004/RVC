"""权重键名归一化 — weight_norm 的两套命名互转。

同一组参数有两种键名：

- 旧版 ``torch.nn.utils.weight_norm``：``<layer>.weight_g``（模长）/ ``<layer>.weight_v``（方向）
- 新版 ``torch.nn.utils.parametrize``：``<layer>.parametrizations.weight.original0`` /
  ``<layer>.parametrizations.weight.original1``

上游 RVC 导出的模型与官方预训练资产都是旧版命名，而本项目模块统一用 parametrize。
``load_state_dict(strict=False)`` 对键名不匹配是静默的，不做改写就会整层跳过——
输出照样能听，但解码器上采样等 208 个参数停在随机初始化，音色与细节全错。
所以凡是从磁盘读进来准备装进模块的权重，都要先过 :func:`to_parametrized`。

为什么在 models 层：这是模型状态的命名约定，不涉及推理/训练流程；
按分层契约 rvc.models(3) 可被 rvc.pipeline(4) 与 rvc.train(5) 依赖。
"""
from collections import OrderedDict

_LEGACY_SUFFIX_G = ".weight_g"
_LEGACY_SUFFIX_V = ".weight_v"
_PARAM_SUFFIX_G = ".parametrizations.weight.original0"
_PARAM_SUFFIX_V = ".parametrizations.weight.original1"


def to_parametrized(state: dict) -> dict:
    """旧版键名改写为 parametrize 键名；已是新命名的原样返回。

    返回值可能与入参同一个对象（无改写时），调用方不要依赖「总是新字典」。
    """
    out = OrderedDict()
    rewritten = 0
    for key, value in state.items():
        if key.endswith(_LEGACY_SUFFIX_G):
            out[key[: -len("weight_g")] + "parametrizations.weight.original0"] = value
            rewritten += 1
        elif key.endswith(_LEGACY_SUFFIX_V):
            out[key[: -len("weight_v")] + "parametrizations.weight.original1"] = value
            rewritten += 1
        else:
            out[key] = value
    return out if rewritten else state


def to_legacy(state: dict) -> dict:
    """parametrize 键名改写为旧版键名；已是旧命名的原样返回。

    写出时用旧版命名，与本仓库已有的模型文件、上游工具保持一致，
    也让 merge_models 能跨来源比较同一层。
    """
    out = OrderedDict()
    rewritten = 0
    for key, value in state.items():
        if key.endswith(_PARAM_SUFFIX_G):
            out[key[: -len("parametrizations.weight.original0")] + "weight_g"] = value
            rewritten += 1
        elif key.endswith(_PARAM_SUFFIX_V):
            out[key[: -len("parametrizations.weight.original1")] + "weight_v"] = value
            rewritten += 1
        else:
            out[key] = value
    return out if rewritten else state


def weight_norm_style(state: dict) -> str:
    """判定字典用的是哪种命名，用于日志诊断。

    返回 ``"legacy"`` / ``"parametrized"`` / ``"mixed"`` / ``"none"``。
    """
    legacy = parametrized = 0
    for key in state:
        if key.endswith(_LEGACY_SUFFIX_G) or key.endswith(_LEGACY_SUFFIX_V):
            legacy += 1
        elif key.endswith(_PARAM_SUFFIX_G) or key.endswith(_PARAM_SUFFIX_V):
            parametrized += 1
    if legacy and parametrized:
        return "mixed"
    if legacy:
        return "legacy"
    if parametrized:
        return "parametrized"
    return "none"
