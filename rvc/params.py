"""运行时参数单例 — GUI 线程写、音频回调线程程读（依赖 GIL 原子性）"""


class Params:
    # 输入门限
    threshold = -60

    # 音调 / 模型参数
    pitch = 0
    index_rate = 0.0
    rms_mix = 0.0
    gender = 0.0
    protect = 0.5
    f0method = "fcpe"

    # 降噪
    I_nr = False
    O_nr = False
    use_pv = False

    # 声学效果
    enable_eq = False
    eq_sub = 0       # 超低频 60Hz
    eq_low = 0       # 低频 200Hz
    eq_mid = 0       # 中频 1kHz
    eq_hi_mid = 0    # 中高频 3kHz
    eq_high = 0      # 高频 8kHz
    reverb = 0

    # 背景音乐
    bgm_enable = False
    bgm_vol = 0.5

    # 副输出
    enable_out2 = False


p = Params()
