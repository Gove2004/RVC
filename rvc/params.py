"""运行时参数单例 — GUI 线程写、音频回调线程程读（依赖 GIL 原子性）"""


class Params:
    # 输入门限
    threshold = -60

    # 音调 / 模型参数
    pitch = 0
    index_rate = 0.0
    rms_mix = 0.0
    gender = 0.0
    f0method = "fcpe"

    # 降噪
    I_nr = False
    O_nr = False
    use_pv = False

    # 声学效果
    enable_eq = False
    eq_low = 0
    eq_mid = 0
    eq_high = 0
    warmth = 0
    compress = 0
    reverb = 0

    # 背景音乐
    bgm_enable = False
    bgm_vol = 0.5

    # 副输出
    enable_out2 = False


p = Params()
