"""环境体检：平台 / 磁盘 / 依赖 / GPU / ffmpeg / 预训练权重 / 数据集。"""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rvc.runtime.paths import CONFIG_ROOT, FFMPEG_EXE, HUBERT_ROOT, PRETRAINED_ROOT, RMVPE_PATH

from autodl import (
    AUDIO_EXTS,
    DATA_ROOT,
    NATIVE_EXTS,
    OPTIONAL_PACKAGES,
    PROJECT_ROOT,
    REQUIRED_PACKAGES,
    EnvFatal,
)
from autodl.logger import TrainLogger
from autodl.prompts import _human_dur, _human_size


def _probe_platform(log: TrainLogger):
    mem_gb = 0.0
    try:
        import os as _os

        if hasattr(_os, "sysconf") and "SC_PHYS_PAGES" in _os.sysconf_names:
            mem_gb = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:
        pass
    log.log(f"系统        : {platform.system()} {platform.release()}")
    log.log(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    log.log(f"CPU / 内存  : {os.cpu_count()} 核 / {mem_gb:.1f} GB" if mem_gb else f"CPU         : {os.cpu_count()} 核")
    log.log(f"项目根目录  : {PROJECT_ROOT}")
    log.log(f"产物根目录  : {DATA_ROOT}")


def _probe_disk(log: TrainLogger):
    for label, path in (("系统盘", PROJECT_ROOT), ("产物盘", DATA_ROOT)):
        try:
            usage = shutil.disk_usage(path if path.exists() else Path(path.anchor))
            log.log(f"{label}剩余   : {_human_size(usage.free)} / {_human_size(usage.total)}  ({path})")
            if usage.free < 5 * 1024 ** 3:
                log.log(f"{label}剩余不足 5GB，训练中途可能写满，建议先清理或扩容", "WARN")
        except Exception as exc:
            log.log(f"{label}空间查询失败: {exc}", "WARN")


def _probe_packages(log: TrainLogger) -> list[str]:
    missing = []
    rows = []
    for import_name, pip_name, desc in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append((pip_name, desc))
            rows.append(f"{pip_name:<14} 缺失  {desc}")
        else:
            rows.append(f"{pip_name:<14} OK")
    for row in rows:
        log.log(f"  {row}")

    if missing:
        log.log("缺少以下必需依赖：", "ERROR")
        for pip_name, desc in missing:
            log.log(f"  · {pip_name:<14} {desc}", "ERROR")
        log.plain()
        log.log("安装命令（AutoDL 镜像一般已带 torch，只补缺的即可）：", "ERROR")
        log.log(f"  pip install {' '.join(p for p, _ in missing)}", "ERROR")
        log.plain()
        log.log("注意：torch 必须与镜像 CUDA 版本匹配，重装前先确认：", "ERROR")
        log.log('  python -c "import torch; print(torch.__version__, torch.version.cuda)"', "ERROR")
        return [p for p, _ in missing]

    absent = [pip for imp, pip, _ in OPTIONAL_PACKAGES if importlib.util.find_spec(imp) is None]
    log.log("必需依赖齐全" + (f"（云端用不到的没装: {', '.join(absent)}）" if absent else ""))
    return []


def _probe_gpu(log: TrainLogger) -> tuple[str, bool, float]:
    """返回 (device, use_fp16, 显存GB)。"""
    import torch

    log.log(f"PyTorch     : {torch.__version__}")
    log.log(f"CUDA 编译   : {torch.version.cuda or '无（CPU 版 torch）'}  |  cuDNN {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        log.log("未检测到可用 CUDA GPU —— 训练会退化到 CPU，速度极慢（不推荐）", "WARN")
        log.log("  检查：nvidia-smi 能否看到显卡；torch 是否为 CUDA 版", "WARN")
        return "cpu", False, 0.0

    idx = 0
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory / 1024 ** 3
    cc = torch.cuda.get_device_capability(idx)
    log.log(f"GPU         : #{idx} {name}")
    log.log(f"显存 / 算力 : {total:.1f} GB / sm_{cc[0]}{cc[1]} / {props.multi_processor_count} SM")

    use_fp16 = True
    if cc[0] < 7:
        log.log("算力 < 7.0（如 P40 / 1080Ti），fp16 收益有限且可能不稳定 → 自动改用 fp32", "WARN")
        use_fp16 = False
    return f"cuda:{idx}", use_fp16, total


def _locate_ffmpeg() -> tuple[str, str]:
    """返回 (路径, 来源)。非 Windows 平台跳过项目内的 Windows exe。"""
    local_exe = FFMPEG_EXE
    env_ffmpeg = os.environ.get("RVC_FFMPEG", "").strip()
    if env_ffmpeg and Path(env_ffmpeg).exists():
        return env_ffmpeg, "环境变量 RVC_FFMPEG"
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg, "系统 PATH"
    if os.name == "nt" and local_exe.exists():
        return str(local_exe), "项目内 assets/ffmpeg/ffmpeg.exe"
    return "", ""


def _probe_ffmpeg(log: TrainLogger) -> str:
    """定位 ffmpeg 并实际验证可执行，返回定位结果（空串 = 未找到）。

    D4：定位结果由调用方显式传参下发（cfg["ffmpeg"]），不再改写
    rvc.io.audio_file 的模块全局。
    """
    chosen, source = _locate_ffmpeg()
    if not chosen:
        log.log("ffmpeg      : 未找到（系统 PATH / assets/ffmpeg 都没有）", "WARN")
        log.log("  素材若含 mp3/m4a/aac 会直接失败，安装：apt install -y ffmpeg", "WARN")
        return ""
    try:
        probe = subprocess.run([chosen, "-version"], capture_output=True, timeout=15)
        if probe.returncode != 0:
            log.log(
                f"ffmpeg      : {chosen} 存在但执行失败（rc={probe.returncode}）"
                " → 请安装系统 ffmpeg：apt install -y ffmpeg",
                "ERROR",
            )
            return ""
    except OSError as exc:
        log.log(f"ffmpeg      : {chosen} 无法执行（{exc}）→ 请安装系统 ffmpeg：apt install -y ffmpeg", "ERROR")
        return ""
    log.log(f"ffmpeg      : {chosen}（来自 {source}）")
    return chosen


def _require_ffmpeg(log: TrainLogger, files: list[Path]):
    """素材里有非原生格式却没有 ffmpeg → 直接失败。"""
    foreign = [p for p in files if p.suffix.lower() not in NATIVE_EXTS]
    if not foreign:
        log.log(f"素材格式    : 全部为 wav/flac/ogg（libsndfile 可直接读取，{len(files)} 个）")
        return
    if _locate_ffmpeg()[0]:
        log.log(f"素材格式    : 含 {len(foreign)} 个需 ffmpeg 解码的文件（如 {foreign[0].name}），已定位 ffmpeg")
        return
    raise EnvFatal(
        f"素材里有 {len(foreign)} 个非 wav/flac/ogg 文件（如 {foreign[0].name}），但没找到 ffmpeg。\n"
        "  安装：apt install -y ffmpeg    （或 conda install -c conda-forge ffmpeg）"
    )


def _probe_assets(log: TrainLogger) -> dict:
    """检查预训练权重与训练配置，返回 {'40k': bool, '48k': bool} 底模可用性。"""
    fatal, warns = [], []

    for sr_k in (40, 48):
        cfg = CONFIG_ROOT / f"{sr_k}ktrain_config.json"
        if not cfg.exists():
            fatal.append(f"训练配置缺失: assets/configs/{sr_k}ktrain_config.json（应随代码仓库一起上传）")

    hubert_root = HUBERT_ROOT
    for variant in ("base", "chinese"):
        hubert_dir = hubert_root / variant
        for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
            path = hubert_dir / f
            if not path.exists():
                fatal.append(f"HuBERT 权重缺失: assets/hubert/{variant}/{f}")
            else:
                log.log(f"  HuBERT {variant:<7} {f:<24} {_human_size(path.stat().st_size)}")

    rmvpe = RMVPE_PATH
    if not rmvpe.exists():
        fatal.append("RMVPE 权重缺失: assets/rmvpe/rmvpe.pt")
    else:
        log.log(f"  RMVPE  rmvpe.pt                {_human_size(rmvpe.stat().st_size)}")

    ready = {}
    for sr_k in (40, 48):
        ok = True
        for key in ("G", "D"):
            path = PRETRAINED_ROOT / f"f0{key}{sr_k}k.pth"
            if not path.exists():
                ok = False
                warns.append(f"底模缺失: assets/pretrained/f0{key}{sr_k}k.pth（{sr_k}k 将从零训练，收敛明显变慢）")
            else:
                log.log(f"  底模   f0{key}{sr_k}k.pth              {_human_size(path.stat().st_size)}")
        ready[f"{sr_k}k"] = ok

    if fatal:
        for msg in fatal:
            log.log(msg, "ERROR")
        log.plain()
        log.log("以上权重都在 .gitignore 里，git clone 不会带下来，需要单独上传：", "ERROR")
        log.log("  · assets/hubert/base/ + assets/hubert/chinese/（各含 config.json + preprocessor_config.json + pytorch_model.bin）", "ERROR")
        log.log("  · assets/rmvpe/rmvpe.pt", "ERROR")
        log.log("  · assets/pretrained/f0G48k.pth + f0D48k.pth（可选，但强烈建议）", "ERROR")
        log.log("  上传方式：AutoDL 网盘 / scp / rsync，放到项目根目录对应位置", "ERROR")
        raise EnvFatal("模型权重缺失")

    for msg in warns:
        log.log(msg, "WARN")
    log.log(f"预训练模型  : {'齐全' if all(ready.values()) else '底模有缺失（见上方 WARN）'}")

    if not any(ready.values()):
        log.log("两个采样率的底模都没有 → 完全从零训练，收敛会慢很多，建议先把权重传上来", "WARN")
    return ready


def _suggest_batch(vram_gb: float, sr_hz: int) -> int:
    if vram_gb >= 24:
        base = 8
    elif vram_gb >= 16:
        base = 6
    elif vram_gb >= 12:
        base = 4
    elif vram_gb >= 8:
        base = 3
    else:
        base = 2
    return base + 2 if sr_hz == 40000 else base


def _scan_audio(input_dir: Path) -> list[Path]:
    """扫描目录下所有音频文件。用 os.walk 替代 rglob（大目录更快）。"""
    if not input_dir.is_dir():
        return []
    result = []
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTS:
                result.append(Path(root) / name)
    return sorted(result)


def _probe_dataset(log: TrainLogger, input_dir: Path, files: list[Path], ffmpeg_path: str | None = None):
    """统计素材：文件数、体积、总时长、采样率分布（超过 300 个只抽样后按比例外推）。"""
    from concurrent.futures import ThreadPoolExecutor

    total_bytes = sum(p.stat().st_size for p in files)
    log.log(f"素材目录    : {input_dir.resolve()}")
    log.log(f"音频文件    : {len(files)} 个，共 {_human_size(total_bytes)}")

    from rvc.io.wav_file import read_audio_metadata

    limit = 300
    sample = files[:limit]
    seconds, sr_dist, scanned = 0.0, {}, 0

    def _probe_one(path: Path):
        try:
            metadata = read_audio_metadata(str(path), ffmpeg_path=ffmpeg_path)
            return metadata["duration"], metadata["samplerate"]
        except Exception:
            return None, None

    with ThreadPoolExecutor(max_workers=min(8, len(sample) or 1)) as pool:
        for dur, sr in pool.map(_probe_one, sample):
            if dur is not None:
                seconds += dur
                sr_dist[sr] = sr_dist.get(sr, 0) + 1
                scanned += 1

    if scanned:
        if scanned < len(files):
            seconds = seconds * len(files) / scanned
        log.log(f"总时长      : 约 {_human_dur(seconds)}" + ("（抽样估算）" if scanned < len(files) else ""))
        log.log("采样率分布  : " + ", ".join(f"{k}Hz×{v}" for k, v in sorted(sr_dist.items())))
    if len(files) < 10:
        log.log("素材文件偏少（<10 个），音质与稳定性会明显下降，建议 30 分钟以上干净人声", "WARN")
    if scanned and seconds < 300:
        log.log("总时长不足 5 分钟，容易过拟合，建议 30 分钟以上", "WARN")
