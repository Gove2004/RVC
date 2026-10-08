#!/usr/bin/env python3
"""有效音域探针 — 连续 F0 扫频 + 单次离线推理 + 逐帧质量分析。

用法:
    python tools/pitch_range_probe.py <input.wav> <model.pth> [options]

流程:
    1. 加载 RVC 模型获取目标采样率
    2. 加载输入人声（自动重复到 8 秒），RMVPE 提取原始 F0
    3. SOLA 连续移调：F0 从 60Hz 平滑扫到 1000Hz（清音帧保持 0）
    4. RVC 离线推理（单次跑完）
    5. RMVPE 提取输出 F0
    6. 按输入 F0 分箱（10Hz/bin），逐箱算 4 项质量指标
    7. 输出有效音域范围 + 质量曲线 CSV
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rvc.core.config import InferenceParams, OfflineParams
from rvc.runtime.paths import RMVPE_PATH, config_path
from rvc.streaming.engine import VoiceEngine

# ── 常量 ──────────────────────────────────────────────
SWEEP_DURATION = 12.0       # 输入音频时长（秒），PSOLA 输出会更短，需留余量
SWEEP_F0_MIN = 60.0         # 扫频起始 F0（Hz）
SWEEP_F0_MAX = 1000.0       # 扫频结束 F0（Hz）
F0_BIN_SIZE = 10.0           # 分析分箱大小（Hz）
FRAME_MS = 4                  # F0 帧长（ms，RMVPE 实际为 4ms/帧 = 250fps）

# 有效范围判据
THRESH_TRACKING = 0.85       # F0 跟踪率（半音尺度相关系数）
THRESH_OCTAVE = 0.05          # 八度错误率
THRESH_JITTER = 0.02          # 抖动（相邻浊音帧相对差均值）
THRESH_VOICING = 0.90         # 清浊一致率
MIN_FRAMES_PER_BIN = 5        # 每箱最少帧数（少于则跳过）


# ── 音频 IO ────────────────────────────────────────────
def load_audio(path, sr):
    """加载音频为 float32 numpy 单声道，重采样到 sr。"""
    from rvc.io.audio_file import load_audio
    wav, _ = load_audio(str(path), sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def save_audio(path, audio, sr):
    """保存 float32 wav。"""
    from rvc.io.wav_file import write_wav
    write_wav(str(path), audio, sr, subtype="FLOAT")


# ── F0 提取（RMVPE 单例复用）──────────────────────────
class F0Extractor:
    """RMVPE F0 提取器，加载一次复用多次。"""

    def __init__(self, device="cuda"):
        from rvc.models.rmvpe import RMVPE
        self.device = device
        self.model = RMVPE(RMVPE_PATH, is_half=(device == "cuda"), device=device)

    def extract(self, audio, sr):
        """提取 F0(Hz) 和 confidence，10ms 帧分辨率。"""
        audio_t = torch.from_numpy(np.ascontiguousarray(audio)).float().to(self.device)
        f0, conf = self.model.infer_from_audio_with_confidence(audio_t, thred=0.05)
        return f0.cpu().numpy().astype(np.float32), conf.cpu().numpy().astype(np.float32)


# ── PSOLA 连续移调 ──────────────────────────────────────
def pitch_shift_psola(audio, sr, orig_f0, target_f0):
    """基音同步重叠相加（PSOLA）连续移调。

    每个浊音段精确取一个基音周期，重采样到目标周期长度后重叠相加。
    与固定帧长的 SOLA 不同，PSOLA 的每段天然是一个周期，极端音高
    偏移（10x 以上）也不会破坏周期结构，RMVPE 可稳定跟踪。

    输出时长会随目标 F0 变化（周期数不变，每周期长度改变）。

    Args:
        audio: 输入音频（float32, 1D）
        sr: 采样率
        orig_f0: 原始 F0 曲线（10ms 帧，Hz，0=清音）
        target_f0: 目标 F0 曲线（10ms 帧，Hz，0=清音）

    Returns:
        移调后音频（时长可能不同）
    """
    output_len = len(audio) + int(sr * 0.2)  # 预留 200ms 余量
    output = np.zeros(output_len, dtype=np.float32)
    in_pos = 0
    out_pos = 0
    unvoiced_hop = int(sr * 0.01)  # 清音帧 10ms

    while in_pos < len(audio) - unvoiced_hop:
        # 当前输入位置对应的 F0 帧索引（RMVPE 4ms/帧）
        f0_idx = min(int(in_pos / (sr * FRAME_MS / 1000)), len(orig_f0) - 1)
        of0 = float(orig_f0[f0_idx])
        tf0 = float(target_f0[f0_idx])

        if of0 > 0 and tf0 > 0:
            period = max(2, int(round(sr / of0)))
            target_period = max(2, int(round(sr / tf0)))
            if in_pos + period > len(audio):
                break

            # 取一个基音周期，加 Hann 窗
            frame = audio[in_pos:in_pos + period].astype(np.float64)
            window = np.hanning(period).astype(np.float64)
            frame = frame * window

            # 重采样到目标周期长度
            shifted = scipy.signal.resample(frame, target_period).astype(np.float32)

            # 重叠相加到输出位置
            out_end = min(out_pos + target_period, output_len)
            output[out_pos:out_end] += shifted[:out_end - out_pos]

            in_pos += period
            out_pos += target_period
        else:
            # 清音：原样复制 10ms
            if in_pos + unvoiced_hop > len(audio):
                break
            out_end = min(out_pos + unvoiced_hop, output_len)
            output[out_pos:out_end] += audio[in_pos:in_pos + (out_end - out_pos)]
            in_pos += unvoiced_hop
            out_pos += unvoiced_hop

    # RMS 归一化
    orig_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-8)
    out_rms = float(np.sqrt(np.mean(output[:out_pos] ** 2)) + 1e-8)
    if out_rms > 1e-8:
        output[:out_pos] *= (orig_rms / out_rms)

    return output[:out_pos]


def generate_sweep_f0(duration):
    """生成线性扫频 F0 曲线（10ms 帧），从 SWEEP_F0_MIN 到 SWEEP_F0_MAX。"""
    n_frames = int(duration * 1000 / FRAME_MS)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    return SWEEP_F0_MIN + t * (SWEEP_F0_MAX - SWEEP_F0_MIN)


# ── 用户配置加载 ────────────────────────────────────────
def load_user_params() -> InferenceParams:
    """从 save_state.json 加载用户推理配置。

    读取 gui 节，构造 InferenceParams（voice/f0/buffer/texture/rms_mix/volume）。
    音频设备配置不加载（离线推理不需要）。
    """
    import json
    state_file = config_path()
    if not state_file.exists():
        print(f"      [警告] 配置文件不存在: {state_file}，使用默认参数")
        return InferenceParams()

    cfg = json.loads(state_file.read_text(encoding="utf-8"))
    gui = cfg.get("gui", {})
    params = InferenceParams()

    # voice
    v = gui.get("voice", {})
    if "formant" in v:
        params.voice.formant = float(v["formant"])
    if "pitch_map_src_min" in v:
        params.voice.pitch_map_src_min = float(v["pitch_map_src_min"])
    if "pitch_map_src_max" in v:
        params.voice.pitch_map_src_max = float(v["pitch_map_src_max"])
    if "pitch_map_dst_min" in v:
        params.voice.pitch_map_dst_min = float(v["pitch_map_dst_min"])
    if "pitch_map_dst_max" in v:
        params.voice.pitch_map_dst_max = float(v["pitch_map_dst_max"])

    # f0
    f = gui.get("f0", {})
    if "method" in f:
        params.f0.method = f["method"]
    if "rmvpe_threshold" in f:
        params.f0.rmvpe_threshold = float(f["rmvpe_threshold"])
    if "fcpe_confidence_threshold" in f:
        params.f0.fcpe_confidence_threshold = float(f["fcpe_confidence_threshold"])

    # buffer
    b = gui.get("buffer", {})
    if "block_time" in b:
        params.buffer.block_time = float(b["block_time"])
    if "crossfade_time" in b:
        params.buffer.crossfade_time = float(b["crossfade_time"])
    if "extra_time" in b:
        params.buffer.extra_time = float(b["extra_time"])

    # texture
    t = gui.get("texture", {})
    if "airflow" in t:
        params.texture.airflow = float(t["airflow"])

    # 顶层
    if "rms_mix" in gui:
        params.rms_mix = float(gui["rms_mix"])
    if "volume" in gui:
        params.volume = float(gui["volume"])

    print(f"      用户配置: formant={params.voice.formant}, "
          f"音域映射={params.voice.pitch_map_src_min:.0f}-{params.voice.pitch_map_src_max:.0f}"
          f"→{params.voice.pitch_map_dst_min:.0f}-{params.voice.pitch_map_dst_max:.0f}, "
          f"F0={params.f0.method}(探针用rmvpe), block={params.buffer.block_time}s")
    return params


# ── 质量分析 ────────────────────────────────────────────
def analyze_quality(input_f0, output_f0):
    """按输入 F0 分箱分析 4 项质量指标。

    Args:
        input_f0: 输入 F0（扫频后的已知 F0，10ms 帧）
        output_f0: 输出 F0（从 RVC 输出提取，10ms 帧）

    Returns:
        dict[bin_center_hz, metrics_dict]
    """
    n_bins = int((SWEEP_F0_MAX - SWEEP_F0_MIN) / F0_BIN_SIZE) + 1
    bins = [{"in": [], "out": [], "in_v": [], "out_v": []} for _ in range(n_bins)]

    min_len = min(len(input_f0), len(output_f0))
    for i in range(min_len):
        if0 = float(input_f0[i])
        of0 = float(output_f0[i])
        if if0 < SWEEP_F0_MIN or if0 > SWEEP_F0_MAX:
            continue
        bin_idx = int((if0 - SWEEP_F0_MIN) / F0_BIN_SIZE)
        if bin_idx >= n_bins:
            continue
        bins[bin_idx]["in"].append(if0)
        bins[bin_idx]["out"].append(of0)
        bins[bin_idx]["in_v"].append(if0 > 0)
        bins[bin_idx]["out_v"].append(of0 > 0)

    results = {}
    for bin_idx, data in enumerate(bins):
        n = len(data["in"])
        if n < MIN_FRAMES_PER_BIN:
            continue
        bin_center = SWEEP_F0_MIN + bin_idx * F0_BIN_SIZE + F0_BIN_SIZE / 2.0

        in_arr = np.array(data["in"], dtype=np.float64)
        out_arr = np.array(data["out"], dtype=np.float64)
        in_v = np.array(data["in_v"], dtype=bool)
        out_v = np.array(data["out_v"], dtype=bool)

        # 清浊一致率（全帧，含清音）
        voicing_agreement = float(np.mean(in_v == out_v))

        # 浊音帧子集（输入和输出都浊音才参与跟踪/抖动分析）
        both_v = in_v & out_v
        if both_v.sum() < 3:
            results[bin_center] = {
                "n_frames": n, "tracking": 0.0, "octave_error": 1.0,
                "jitter": 1.0, "voicing_agreement": voicing_agreement, "valid": False,
            }
            continue

        in_v_arr = in_arr[both_v]
        out_v_arr = out_arr[both_v]

        # 1. F0 跟踪率（半音尺度 Pearson 相关）
        in_midi = 69.0 + 12.0 * np.log2(np.clip(in_v_arr, 1e-6, None) / 440.0)
        out_midi = 69.0 + 12.0 * np.log2(np.clip(out_v_arr, 1e-6, None) / 440.0)
        if np.std(in_midi) > 0.05 and np.std(out_midi) > 0.05:
            tracking = float(np.corrcoef(in_midi, out_midi)[0, 1])
            if np.isnan(tracking):
                tracking = 0.0
        else:
            tracking = 0.0

        # 2. 八度错误率（输出 ≈ 2× 或 0.5× 输入）
        ratio = out_v_arr / (in_v_arr + 1e-8)
        octave_err = float(np.mean(
            (np.abs(ratio - 2.0) < 0.3) | (np.abs(ratio - 0.5) < 0.15)
        ))

        # 3. 抖动（相邻浊音帧 F0 相对差均值，按时间顺序）
        if len(out_v_arr) > 1:
            jitter = float(np.mean(np.abs(np.diff(out_v_arr)) / (out_v_arr[:-1] + 1e-8)))
        else:
            jitter = 1.0

        valid = (
            tracking >= THRESH_TRACKING
            and octave_err <= THRESH_OCTAVE
            and jitter <= THRESH_JITTER
            and voicing_agreement >= THRESH_VOICING
        )

        results[bin_center] = {
            "n_frames": n,
            "tracking": tracking,
            "octave_error": octave_err,
            "jitter": jitter,
            "voicing_agreement": voicing_agreement,
            "valid": valid,
        }

    return results


# ── 结果输出 ────────────────────────────────────────────
def print_results(results):
    """打印有效音域分析结果。"""
    if not results:
        print("\n无有效数据（所有分箱帧数不足）")
        return

    valid_f0s = sorted(f0 for f0, r in results.items() if r["valid"])

    print("\n" + "=" * 72)
    print("有效音域分析结果")
    print("=" * 72)

    if valid_f0s:
        # 找最长连续有效区间
        ranges = []
        start = valid_f0s[0]
        prev = valid_f0s[0]
        for f0 in valid_f0s[1:]:
            if f0 - prev > F0_BIN_SIZE * 1.5:
                ranges.append((start, prev))
                start = f0
            prev = f0
        ranges.append((start, prev))

        print(f"\n有效音域（{len(valid_f0s)} 个分箱 / 共 {len(results)} 个）:")
        for lo, hi in ranges:
            print(f"  {lo:.0f}Hz ~ {hi:.0f}Hz")
    else:
        print("\n未找到完全满足所有判据的音域范围")

    # 详细表格
    print(f"\n{'F0(Hz)':>8} {'帧数':>6} {'跟踪率':>8} {'八度错':>8} {'抖动':>8} {'清浊':>8} {'有效':>6}")
    print("-" * 72)
    for f0 in sorted(results.keys()):
        r = results[f0]
        print(f"{f0:>8.0f} {r['n_frames']:>6} {r['tracking']:>8.3f} "
              f"{r['octave_error']:>8.3f} {r['jitter']:>8.4f} {r['voicing_agreement']:>8.3f} "
              f"{'✓' if r['valid'] else '✗':>6}")

    # 各指标单独边界
    print("\n各指标边界（从低到高扫描，首次不达标处）:")
    for metric, label, threshold, higher_better in [
        ("tracking", "跟踪率", THRESH_TRACKING, True),
        ("octave_error", "八度错误", THRESH_OCTAVE, False),
        ("jitter", "抖动", THRESH_JITTER, False),
        ("voicing_agreement", "清浊一致", THRESH_VOICING, True),
    ]:
        boundary = None
        for f0 in sorted(results.keys()):
            val = results[f0][metric]
            bad = (val < threshold) if higher_better else (val > threshold)
            if bad:
                boundary = f0
                break
        if boundary:
            print(f"  {label}: {boundary:.0f}Hz 处首次不达标（阈值={threshold}）")
        else:
            print(f"  {label}: 全程达标")


def save_csv(results, output_path):
    """保存质量曲线到 CSV。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("f0_hz,n_frames,tracking,octave_error,jitter,voicing_agreement,valid\n")
        for f0 in sorted(results.keys()):
            r = results[f0]
            f.write(f"{f0:.1f},{r['n_frames']},{r['tracking']:.4f},"
                    f"{r['octave_error']:.4f},{r['jitter']:.6f},{r['voicing_agreement']:.4f},"
                    f"{1 if r['valid'] else 0}\n")


# ── 主流程 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RVC 模型有效音域探针")
    parser.add_argument("input", help="输入人声文件（wav/mp3/m4a 等）")
    parser.add_argument("model", help="RVC 模型文件（.pth）")
    parser.add_argument("--hubert", default="chinese", help="HuBERT 变体（默认 chinese）")
    parser.add_argument("--duration", type=float, default=SWEEP_DURATION,
                        help=f"扫频时长秒（默认 {SWEEP_DURATION}）")
    parser.add_argument("--output", default=None,
                        help="输出 CSV 路径（默认 pitch_range_result.csv）")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留临时扫频输入/输出 wav（默认清理）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    temp_in = PROJECT_ROOT / "temp_probe_input.wav"
    temp_out = PROJECT_ROOT / "temp_probe_output.wav"

    # ── 1. 加载用户配置 + RVC 模型 ──
    print(f"[1/7] 加载用户配置 + RVC 模型: {args.model}")
    user_params = load_user_params()
    engine = VoiceEngine(user_params)
    target_sr = engine.load_model(args.model, hubert=args.hubert)
    print(f"      模型目标采样率: {target_sr}Hz")

    # ── 2. 加载输入音频，自动重复到扫频时长 ──
    print(f"\n[2/7] 加载输入音频: {args.input}")
    audio = load_audio(args.input, target_sr)
    print(f"      原始时长: {len(audio)/target_sr:.1f}s")
    target_samples = int(args.duration * target_sr)
    if len(audio) < target_samples:
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_samples]
        print(f"      自动重复到 {args.duration}s")
    else:
        audio = audio[:target_samples]

    # ── 3. 提取原始 F0 ──
    print(f"\n[3/7] 提取原始 F0（RMVPE, {device}）...")
    f0_ext = F0Extractor(device)
    orig_f0, orig_conf = f0_ext.extract(audio, target_sr)
    print(f"      帧数: {len(orig_f0)}, 浊音帧: {(orig_f0 > 0).sum()} "
          f"({100*(orig_f0>0).mean():.0f}%)")

    # ── 4. 生成扫频 F0 + 连续移调 ──
    print(f"\n[4/7] 连续移调（{SWEEP_F0_MIN:.0f}Hz → {SWEEP_F0_MAX:.0f}Hz, {args.duration}s）...")
    sweep_f0 = generate_sweep_f0(args.duration)
    # 对齐长度，清音帧保持 0
    min_len = min(len(orig_f0), len(sweep_f0))
    target_f0 = np.zeros(min_len, dtype=np.float32)
    voiced = orig_f0[:min_len] > 0
    target_f0[voiced] = sweep_f0[:min_len][voiced]

    swept_audio = pitch_shift_psola(audio, target_sr, orig_f0[:min_len], target_f0)
    save_audio(temp_in, swept_audio, target_sr)
    print(f"      扫频输入已保存: {temp_in}（时长 {len(swept_audio)/target_sr:.1f}s）")

    # 从移调后音频实际提取 F0（PSOLA 输出时长会变，且实际 F0 可能偏离理论值）
    print(f"      验证移调后 F0...")
    swept_f0, swept_conf = f0_ext.extract(swept_audio, target_sr)
    voiced_pct = 100 * (swept_f0 > 0).mean()
    f0_min = swept_f0[swept_f0 > 0].min() if (swept_f0 > 0).any() else 0
    f0_max = swept_f0[swept_f0 > 0].max() if (swept_f0 > 0).any() else 0
    print(f"      移调后 F0: {len(swept_f0)} 帧, 浊音 {voiced_pct:.0f}%, "
          f"范围 {f0_min:.0f}~{f0_max:.0f}Hz")

    # ── 5. RVC 离线推理（用户配置，但音域映射设恒等以测模型原始响应）──
    print(f"\n[5/7] RVC 离线推理（用户配置 + 恒等音域映射）...")
    task = OfflineParams(
        input_path=str(temp_in),
        output_path=str(temp_out),
        model_path=str(args.model),
        hubert=args.hubert,
    )
    # 从用户配置复制所有推理参数
    task.voice = user_params.voice
    task.f0 = user_params.f0
    task.buffer = user_params.buffer
    task.texture = user_params.texture
    task.rms_mix = user_params.rms_mix
    task.volume = user_params.volume
    # 探针内部用 RMVPE（对 PSOLA 移调音频跟踪更稳定）
    # 用户配置的 FCPE 对移调音频多数频段提取为 0，导致测不出模型真实能力
    task.f0.method = "rmvpe"
    # 音域映射设恒等：src=dst，扫频 F0 直接送入合成器，测模型原始有效范围
    task.voice.pitch_map_src_min = SWEEP_F0_MIN
    task.voice.pitch_map_src_max = SWEEP_F0_MAX
    task.voice.pitch_map_dst_min = SWEEP_F0_MIN
    task.voice.pitch_map_dst_max = SWEEP_F0_MAX
    engine.process_file(task)
    print(f"      推理输出已保存: {temp_out}")

    # ── 6. 提取输出 F0 + 质量分析（按原始扫频 F0 分箱）──
    print(f"\n[6/7] 提取输出 F0 并分析质量...")
    output_audio = load_audio(temp_out, target_sr)
    output_f0, output_conf = f0_ext.extract(output_audio, target_sr)

    # 用实际移调 F0 作为分析基准（恒等映射下 = 合成器收到的 F0）
    min_len = min(len(swept_f0), len(output_f0))
    results = analyze_quality(swept_f0[:min_len], output_f0[:min_len])
    print_results(results)

    # ── 7. 保存 CSV + 清理 ──
    csv_path = Path(args.output) if args.output else PROJECT_ROOT / "pitch_range_result.csv"
    save_csv(results, csv_path)
    print(f"\n[7/7] 质量曲线已保存: {csv_path}")

    if not args.keep_temp:
        temp_in.unlink(missing_ok=True)
        temp_out.unlink(missing_ok=True)
        print("临时文件已清理")
    else:
        print(f"临时文件已保留: {temp_in}, {temp_out}")

    # 释放
    del f0_ext
    del engine
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
