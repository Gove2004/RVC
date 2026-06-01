"""运行时参数单例"""


class Params:
    threshold = -60; pitch = 0; index_rate = 0.0; rms_mix = 0.0; gender = 0.0
    f0method = "fcpe"; I_nr = False; O_nr = False; use_pv = False
    enable_eq = False; eq_low = 0; eq_mid = 0; eq_high = 0
    warmth = 0; compress = 0; reverb = 0
    bgm_enable = False; bgm_vol = 0.5
    enable_out2 = False


p = Params()
