"""实时效果参数同步与处理。"""
import torch


def apply_pre_sola_effects(
    audio: torch.Tensor,
    params,
    eq,
    last_eq_params,
) -> tuple[torch.Tensor, tuple | None]:
    if not params.enable_eq or not eq:
        return audio, last_eq_params

    current_eq = (params.eq_sub, params.eq_low, params.eq_mid, params.eq_hi_mid, params.eq_high)
    if last_eq_params != current_eq:
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
    if not reverb or params.reverb <= 0:
        return audio, last_reverb_mix

    if last_reverb_mix != params.reverb:
        reverb.set_mix(params.reverb)
        last_reverb_mix = params.reverb
    return reverb(audio), last_reverb_mix
