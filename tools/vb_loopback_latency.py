"""VB-CABLE 环回延迟实测（按 host API 分别测）。

往 CABLE Input 打一串噪声脉冲，同时从 CABLE Output 录回，用归一化互相关找
脉冲到达位置，减掉写入时刻 = 写入到读出的真实经过时间。相关峰与包络起点两
个独立口径应当一致，不一致说明录音被别的声音污染。

同一对虚拟线在 MME / DirectSound / WASAPI 下是三个不同的设备索引，固定开销
差别极大，逐个测；采样率用该设备默认值。WDM-KS 下设备名是 "VB-Audio Point"，
不含 cable，不在这里匹配（KS 是独占模式，双工会直接失败）。
GUI 存档里选的是哪个驱动看 audio.hostapi。

用法（先在软件里停止转换，否则变声输出会串进录音）：
    python tools/vb_loopback_latency.py [驱动名子串] [块秒数逗号分隔] [重复次数]
    python tools/vb_loopback_latency.py WASAPI 0.02,0.05 3
"""
import sys

import numpy as np
import sounddevice as sd

DUR = 3.0          # 每次录音时长（秒）
BURST_S = 0.05     # 脉冲串长度
ONSET_S = 0.4      # 脉冲在播放缓冲里的起点
SEARCH_S = 0.9     # 最大搜索延迟
REPEATS = 3
BLOCKS_S = (0.01, 0.02, 0.05, 0.20)


def pairs_by_hostapi():
    """{hostapi 名: (输入索引, 输出索引)}，只收虚拟线成对出现的驱动。"""
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    found = {}
    for i, d in enumerate(devs):
        name = d["name"].lower()
        api = apis[d["hostapi"]]["name"]
        if "cable output" in name and d["max_input_channels"] > 0:
            found.setdefault(api, [None, None])[0] = i
        elif "cable input" in name and d["max_output_channels"] > 0:
            found.setdefault(api, [None, None])[1] = i
    return {k: v for k, v in found.items() if all(x is not None for x in v)}


def peaks(r, sr, offset_samples=0, top=5, min_r=0.30):
    """相关曲线上最高的几个局部峰（写入到读出的延迟 ms, r），区分直达与回声。"""
    cands = [((k - offset_samples) * 1000.0 / sr, float(r[k])) for k in range(2, len(r) - 2)
             if r[k] >= min_r and r[k] == max(r[k - 2:k + 3])]
    cands.sort(key=lambda p: -p[1])
    kept = []
    for ms, v in cands:
        if all(abs(ms - k[0]) > 20 for k in kept):
            kept.append((ms, v))
        if len(kept) == top:
            break
    return [(round(ms, 1), v) for ms, v in kept]


def onset_ms(x, sr, thr=0.05):
    """首个超过阈值的样本时刻（减掉 ONSET_S），作为相关峰的第二意见。"""
    idx = int(np.argmax(np.abs(x) > thr * np.abs(x).max())) if np.abs(x).max() > 0 else -1
    return (idx / sr - ONSET_S) * 1000.0 if idx >= 0 else float("nan")


def one(sr: int, blocksize: int, in_idx: int, out_idx: int):
    ch = 1
    rng = np.random.default_rng(0)
    burst = rng.standard_normal(int(BURST_S * sr)).astype(np.float32)
    taper = np.hanning(64).astype(np.float32)
    burst[:32] *= taper[:32]
    burst[-32:] *= taper[-32:]

    n_total = int(DUR * sr)
    onset = int(ONSET_S * sr)
    play = np.zeros((n_total, ch), dtype=np.float32)
    play[onset:onset + burst.size, 0] = burst * 0.7

    rec = np.zeros((n_total, ch), dtype=np.float32)
    state = {"i": 0}

    def cb(indata, outdata, frames, t, status):
        i = state["i"]
        if i + frames <= n_total:
            outdata[:] = play[i:i + frames]
            rec[i:i + frames] = indata
        else:
            outdata[:] = 0.0
        state["i"] = i + frames

    with sd.Stream(device=(in_idx, out_idx), samplerate=sr, channels=ch,
                   dtype="float32", blocksize=blocksize, latency="low",
                   callback=cb):
        sd.sleep(int(DUR * 1000) + 300)

    x = rec[:, 0]
    floor = float(np.sqrt(np.mean(x[: onset // 2] ** 2)))  # 脉冲前的底噪

    n = x.size
    X = np.fft.rfft(x, n)
    B = np.fft.rfft(np.pad(burst, (0, n - burst.size)), n)
    corr = np.fft.irfft(X * np.conj(B), n)[: int(SEARCH_S * sr)]
    e_b = float(np.sqrt(np.sum(burst ** 2)))
    win = np.lib.stride_tricks.sliding_window_view(x, burst.size)
    e_x = np.sqrt((win[: corr.size] ** 2).sum(axis=1))
    # 相关值只在「窗内确有信号」时才有意义：大片静音会让 FFT 数值泄漏除以近零
    # 能量后放大成假峰（实测出现过 r=250），所以按模板能量的 10% 设门限。
    r = corr / (e_b * e_x + 1e-12)
    r[e_x < 0.1 * e_b] = -1.0

    lag = int(np.argmax(r))
    # lag 是脉冲在录音时间轴上的**位置**；它在 ONSET_S 时刻就被写进播放缓冲，
    # 所以写入到读出的延迟 = 位置 − 写入时刻。
    return (lag - onset) / sr * 1000.0, float(r[lag]), floor, peaks(r, sr, onset), onset_ms(x, sr)


pairs = pairs_by_hostapi()
if len(sys.argv) > 1:  # 只测名字里含该子串的驱动，如 WASAPI
    key = sys.argv[1].lower()
    pairs = {k: v for k, v in pairs.items() if key in k.lower()}
blocks = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(BLOCKS_S))
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else REPEATS
if not pairs:
    raise SystemExit("没找到虚拟线设备对，先确认 VB-CABLE 已安装。")

devs = sd.query_devices()
print(f"{'驱动':<20}{'采样率':>7}  环回延迟（中位 / 三次，ms）")
for api, (in_idx, out_idx) in sorted(pairs.items()):
    sr = int(devs[out_idx]["default_samplerate"])
    low_out = devs[out_idx]["default_low_output_latency"] * 1000
    low_in = devs[in_idx]["default_low_input_latency"] * 1000
    print(f"\n{api:<20}{sr:>6}  驱动自称 low: out={low_out:.0f}ms in={low_in:.0f}ms")
    pts = []
    for bt in blocks:
        bs = max(1, int(round(bt * sr)))
        try:
            res = [one(sr, bs, in_idx, out_idx) for _ in range(repeats)]
        except Exception as e:  # 某些驱动不支持双工，如实报出来
            print(f"  块 {bt:.2f}s  测不了：{type(e).__name__}: {e}")
            continue
        ms = [r[0] for r in res]
        med = float(np.median(ms))
        conf = res[-1][1]
        if conf < 0.3:
            print(f"  块 {bt:.2f}s (bs={bs:5d}) 中位={med:7.1f}  "
                  f"各次=[{' '.join(f'{v:.1f}' for v in ms)}]  ← 相关仅 {conf:.2f}，峰不可信")
            continue
        pts.append((bt, med))
        floor, pk, on = res[-1][2], res[-1][3], res[-1][4]
        print(f"  块 {bt:.2f}s (bs={bs:5d}) 中位={med:7.1f}  "
              f"各次=[{' '.join(f'{v:.1f}' for v in ms)}]  相关={conf:.3f} 底噪={floor:.5f} "
              f"起点={on:.1f}ms\n      相关峰: {pk}")
    if len(pts) >= 2:
        slope, intercept = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        print(f"      拟合: 延迟 ≈ {intercept:.0f}ms 固定 + {slope / 1000:.2f} × 块时长")
