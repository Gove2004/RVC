"""实时效果参数同步与处理。

负责在音频回调中动态更新效果器参数（EQ均衡器增益、混响混合比例），
支持参数变化检测以避免不必要的重计算。

处理流程：
1. 预SOLA效果（EQ）：检查EQ参数是否变化，变化则更新ParametricEQ频段，应用均衡
2. 后SOLA效果（混响）：检查混响参数是否变化，变化则更新SimpleReverb混合比例，应用混响
两个效果都采用缓存模式，只在参数实际改变时重新计算效果器设置。
"""
import torch


def apply_pre_sola_effects(
    audio: torch.Tensor,
    params,
    eq,
    last_eq_params,
) -> tuple[torch.Tensor, tuple | None]:
    """应用SOLA前的效果（主要为EQ均衡器）。

    参数变化检测：若当前EQ参数与上次不同，则更新EQ各频段增益并缓存新参数。
    若EQ禁用或无EQ对象，则直接返回原音频和缓存参数。

    Args:
        audio: 待处理的音频张量 [samples]
        params: 包含enable_eq及各频段增益的参数对象
        eq: ParametricEQ效果器实例，可为None
        last_eq_params: 上一次设置的eq参数元组，用于变化检测

    Returns:
        (处理后音频, 更新的last_eq_params)
    """
    if not params.enable_eq or not eq:
        return audio, last_eq_params

    current_eq = (params.eq_sub, params.eq_low, params.eq_mid, params.eq_hi_mid, params.eq_high)
    if last_eq_params is None or last_eq_params != current_eq:
        eq.set_band("sub", params.eq_sub)
        eq.set_band("low", params.eq_low)
        eq.set_band("mid", params.eq_mid)
        eq.set_band("hi_mid", params.eq_hi_mid)
        eq.set_band("high", params.eq_high)
        last_eq_params = current_eq
    return eq(audio), last_eq_params


def apply_post_sola_effects(
    audio: torch.Tensor,
    params,
    reverb,
    last_reverb_mix,
) -> tuple[torch.Tensor, float | None]:
    """应用SOLA后的效果（主要为混响）。

    参数变化检测：若当前混响混合比例与上次不同，则更新Reverb混合值并缓存。
    若混响禁用或混合比例为0，则直接返回原音频。

    Args:
        audio: 待处理的音频张量 [samples]
        params: 包含reverb参数的对象
        reverb: SimpleReverb效果器实例，可为None
        last_reverb_mix: 上一次的混响混合比例

    Returns:
        (处理后音频, 更新的last_reverb_mix)
    """
    if not reverb or params.reverb <= 0:
        return audio, last_reverb_mix

    if last_reverb_mix != params.reverb:
        reverb.set_mix(params.reverb)
        last_reverb_mix = params.reverb
    return reverb(audio), last_reverb_mix
