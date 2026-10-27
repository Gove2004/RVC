"""权重键名归一化 — weight_norm 的两套命名互转。

同一组参数有两种键名：

- 旧版 ``torch.nn.utils.weight_norm``：``<layer>.weight_g``（模长）/ ``<layer>.weight_v``（方向）
- 新版 ``torch.nn.utils.parametrize``：``<layer>.parametrizations.weight.original0`` /
  ``<layer>.parametrizations.weight.original1``

上游 RVC 导出的模型与官方预训练资产都是旧版命名，本项目模块统一用
``parametrizations.weight_norm``。

**加载不需要改写**：``ParametrizationList._load_from_state_dict`` 自带旧键名兼容层，
``load_state_dict`` 能吃 ``weight_g``/``weight_v`` 并正确落到 ``original0``/``original1``
（实测 missing/unexpected 均为空、物化权重与源文件一致）。别因为
``state_dict()`` 里看不到 ``weight_g`` 就以为权重没加载——那是键名比对，
不是加载结果。

**需要改写的是「两份字典互相比对」的场合**，那里没有 torch 兼容层：

- ``merge_models``：按裸键名取交集来加权平均。一个旧命名文件 + 一个本流程导出的
  文件会让同一层变成两个不同的键，权重不会平均，合并结果还同时带两套键名，
  再加载时行为取决于 dict 顺序
- ``export_model``：落盘统一用旧版命名，与本仓库已有模型文件、上游工具一致

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

    只用于「两份字典互相比对」的场合（如 merge_models 取交集），
    加载进模块不需要它——torch 的 ParametrizationList 自带旧键名兼容层。

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
