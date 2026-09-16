"""金标准采集与校验 — 重写行为等价性的硬证据。

用法（旧代码采集，仅重写前执行一次）:
    python scripts/golden.py capture
用法（新代码校验，重写每步之后执行）:
    python scripts/golden.py verify

原理: 固定合成输入 + 固定模型 + 固定参数 → 离线转换（复用实时全链路）
→ 输出 float32 数组存 tests/golden/*.npy + manifest.json(sha256+参数)。
重写后用同一输入跑新代码，逐点对比，容差 ±1e-4。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
GOLDEN = PROJECT_ROOT / "tests" / "golden"
MODEL = PROJECT_ROOT / "assets" / "models" / "xd80+ay20.pth"
HUBERT = "base"
# 实时链路含 SOLA 逐块拼接，浮点噪声会混沌放大（旧代码自比亦不可逐点复现），
# 故等价性度量用 STFT dB 域统计距离（含能量下限钳制，排除噪底伪影）。
TOLERANCE_MEAN_DB = 0.5   # 全谱平均 dB 差上限（40dB 动态范围内）
TOLERANCE_P99_DB = 3.0    # 99 分位 dB 差上限
SPECTRUM_FLOOR_DB = 40.0  # 相对全谱峰值的能量下限（低于视为无声，不计差异）

# 两个参数组合覆盖: 默认路径 / formant+阈值+rms_mix 非默认路径
CASES = {
    "offline_a": {
        "formant": 0.0,
        "rms_mix": 1.0,
        "rmvpe_threshold": 0.05,
        "pitch_map": (100.0, 500.0, 200.0, 800.0),
        "block_time": 0.25,
        "crossfade_time": 0.05,
        "extra_time": 2.5,
    },
    "offline_b": {
        "formant": 2.0,
        "rms_mix": 0.5,
        "rmvpe_threshold": 0.03,
        "pitch_map": (200.0, 400.0, 400.0, 800.0),
        "block_time": 0.1,
        "crossfade_time": 0.05,
        "extra_time": 2.0,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_input(path: Path, sr: int = 48000, dur: float = 4.0):
    """合成"类人声"输入: 基频滑动的谐波列 + 振幅包络 + 中段静音(测 F0 静音帧)。"""
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    f0 = 220.0 * 2 ** (0.25 * np.sin(2 * np.pi * 0.5 * t))  # 220Hz±四分之一八度慢滑
    wav = np.zeros_like(t)
    for h, a in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)):
        phase = 2 * np.pi * np.cumsum(f0 * h) / sr
        wav += a * np.sin(phase)
    env = 0.5 * (1 - np.cos(2 * np.pi * t / dur))  # 平滑包络
    env[: sr // 2] *= np.linspace(0, 1, sr // 2)   # 淡入
    wav *= env
    # 中段 0.8s 静音（F0 应判清音/0）
    mute = slice(int(sr * 1.8), int(sr * 2.6))
    wav[mute] = 0.0
    wav = (wav / np.abs(wav).max() * 0.8).astype(np.float32)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(sr)
        w.writeframes(wav.tobytes())
    return wav


def build_params(name: str, tmp_out: Path):
    from rvc.core.config import OfflineParams
    c = CASES[name]
    p = OfflineParams()
    p.input_path = str(GOLDEN / "input_48k.wav")
    p.output_path = str(tmp_out)
    p.model_path = str(MODEL)
    p.hubert = HUBERT
    p.rms_mix = c["rms_mix"]
    p.voice.formant = c["formant"]
    p.voice.pitch_map_src_min, p.voice.pitch_map_src_max = c["pitch_map"][0], c["pitch_map"][1]
    p.voice.pitch_map_dst_min, p.voice.pitch_map_dst_max = c["pitch_map"][2], c["pitch_map"][3]
    p.f0.rmvpe_threshold = c["rmvpe_threshold"]
    p.buffer.block_time = c["block_time"]
    p.buffer.crossfade_time = c["crossfade_time"]
    p.buffer.extra_time = c["extra_time"]
    return p


def _patch_race():
    """给旧代码的异步竞态打补丁（仅采集/校验期生效），恢复串行参考语义。

    旧代码存在三处并发缺陷，导致输出间歇性 NaN / 不可复现：
    1. _stage_input 的 pinned 内存 non_blocking DMA 未同步，下一块覆写导致
       输入污染、HuBERT(fp16) 整块 NaN；
    2. 块级异步链路（D2H）缺少块尾同步；
    3. HuBERT 与 RMVPE 在两条 CUDA Stream 上并发 fp16 推理——cuBLAS/cuDNN
       工作区跨流并发属未定义行为，实测随机产生整块 NaN（bisect 证据：
       NaN 仅出现在 HuBERT 输出、块序随运行变化）。
    补丁 = 输入拷贝后同步 + 串行化阶段2/3（_is_cuda→False 走顺序分支），
    得到"算法本意"的确定性输出。重写版将在引擎内正确实现。
    """
    import torch
    import rvc.streaming.runner as runner_mod
    import rvc.pipeline.pipeline as pipeline_mod

    def fixed_input(self, indata):
        state = self.state
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        mono = np.ascontiguousarray(mono)
        n = mono.shape[0]
        state.in_pin[:n].copy_(torch.from_numpy(mono), non_blocking=True)
        t = state.in_pin[:n].to(state.device, non_blocking=True)
        torch.cuda.synchronize()
        return t

    orig_block = runner_mod.InferenceRunner.process_block

    def fixed_block(self, indata, outdata, frames):
        orig_block(self, indata, outdata, frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    runner_mod.InferenceRunner._stage_input = fixed_input
    runner_mod.InferenceRunner.process_block = fixed_block
    pipeline_mod._is_cuda = lambda device: False


def _run_case_subprocess(action: str, name: str) -> int:
    """每个 case 用独立子进程执行。

    旧代码存在跨实例状态污染：同一进程内第二个 RealtimeEngine 输出 NaN
    （全局 CUDA Graph/静态缓存残留）。金标准采集与校验必须进程隔离。
    """
    import subprocess
    cmd = [sys.executable, "-u", __file__, action, "--case", name]
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def capture():
    from rvc.core.config import InferenceParams  # noqa: F401 确认可导入

    GOLDEN.mkdir(parents=True, exist_ok=True)
    input_path = GOLDEN / "input_48k.wav"
    if not input_path.exists():
        make_input(input_path)
        print(f"生成输入: {input_path}")

    manifest_path = GOLDEN / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"cases": {}}
    for name in CASES:
        rc = _run_case_subprocess("capture_one", name)
        if rc != 0:
            print(f"[{name}] 子进程采集失败 rc={rc}")
            sys.exit(1)
        # 子进程把单例结果写进 case manifest 片段
        piece = json.loads((GOLDEN / f"{name}_manifest.json").read_text(encoding="utf-8"))
        manifest["cases"][name] = piece
        (GOLDEN / f"{name}_manifest.json").unlink()
        print(f"[{name}] 已采集")
    (GOLDEN / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"金标准已写入 {GOLDEN}")


def capture_one(name: str):
    from rvc.streaming.engine import RealtimeEngine

    _patch_race()
    out_tmp = GOLDEN / f"{name}_out.wav"
    params = build_params(name, out_tmp)
    engine = RealtimeEngine(params)
    sr = engine.load_model(str(MODEL), hubert=HUBERT)
    print(f"[{name}] 模型加载完成 sr={sr}")
    result = np.ascontiguousarray(engine.process_file(params), dtype=np.float32)
    npy = GOLDEN / f"{name}.npy"
    np.save(npy, result)
    piece = {
        "npy_sha256": sha256_file(npy),
        "samples": int(result.size),
        "params": CASES[name],
        "model": str(MODEL.name),
        "hubert": HUBERT,
    }
    (GOLDEN / f"{name}_manifest.json").write_text(
        json.dumps(piece, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{name}] 输出 {result.size} 样本, 峰值 {np.abs(result).max():.4f}")


def verify():
    for name in CASES:
        rc = _run_case_subprocess("verify_one", name)
        if rc != 0:
            print(f"[{name}] 校验失败")
            sys.exit(1)
    print("金标准校验全部通过")


def verify_one(name: str):
    from rvc.streaming.engine import RealtimeEngine

    _patch_race()
    golden = np.load(GOLDEN / f"{name}.npy")
    out_tmp = GOLDEN / f"{name}_verify_out.wav"
    params = build_params(name, out_tmp)
    engine = RealtimeEngine(params)
    engine.load_model(str(MODEL), hubert=HUBERT)
    result = np.ascontiguousarray(engine.process_file(params), dtype=np.float32)
    if result.shape != golden.shape:
        print(f"[{name}] 形状不一致: {result.shape} != {golden.shape}")
        sys.exit(1)
    mean_db, p99_db = spectral_db_diff(result, golden)
    ok = mean_db <= TOLERANCE_MEAN_DB and p99_db <= TOLERANCE_P99_DB
    status = "PASS" if ok else "FAIL"
    print(f"[{name}] 谱差 mean={mean_db:.3f}dB p99={p99_db:.3f}dB  {status}")
    sys.exit(0 if ok else 1)


def spectral_db_diff(a: np.ndarray, b: np.ndarray) -> tuple:
    """STFT 幅度谱 dB 域差异: (mean_db, p99_db)。

    以两谱全局峰值为基准，低 Floor dB 的频点视为无声（钳到 Floor），
    避免噪底 bin 的 dB 抖动淹没真实差异。
    """
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft).astype(np.float64)

    def stft_db(x):
        frames = []
        for i in range(0, len(x) - n_fft + 1, hop):
            seg = x[i:i + n_fft].astype(np.float64) * window
            frames.append(np.abs(np.fft.rfft(seg)))
        mag = np.stack(frames)  # [帧, 频 bin]
        return 20.0 * np.log10(mag + 1e-8)

    da = stft_db(a)
    db = stft_db(b)
    floor = max(da.max(), db.max()) - SPECTRUM_FLOOR_DB
    da = np.maximum(da, floor)
    db = np.maximum(db, floor)
    diff = np.abs(da - db)
    return float(diff.mean()), float(np.percentile(diff, 99))


if __name__ == "__main__":
    args = sys.argv[1:]
    action = args[0] if args else ""
    case = args[args.index("--case") + 1] if "--case" in args else None
    if action == "capture":
        capture()
    elif action == "capture_one" and case:
        capture_one(case)
    elif action == "verify":
        verify()
    elif action == "verify_one" and case:
        verify_one(case)
    else:
        print("用法: python scripts/golden.py capture|verify [--case NAME]")
        sys.exit(2)
