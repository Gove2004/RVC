"""RVC 云 GPU（AutoDL / 恒源云 / 揽睿星舟等）训练向导。

交互式，无需记任何参数：
    python autodl_train.py

流程：
    1) 检查 Python 库环境（依赖 / GPU / 磁盘 / ffmpeg）
    2) 检查预训练模型（HuBERT / RMVPE / 底模 / 训练配置）
    3) 训练参数追问（全部带默认值，一路回车即可）
    4) 开始训练（预处理 → F0 → 特征 → 训练，已完成步骤自动跳过）

续训规则：检测到 checkpoint 就自动接着训；要重新开始，直接删掉
/root/autodl-tmp/logs/<实验名>/4_checkpoints 即可。

产物默认落在数据盘（云上 = /root/autodl-tmp，避免撑爆 30G 系统盘）：
    日志/切片/checkpoint : /root/autodl-tmp/logs/<实验名>/
    导出模型             : /root/autodl-tmp/models/<实验名>/
可用环境变量 RVC_OUT_ROOT 改输出根目录，RVC_FFMPEG 指定 ffmpeg。

自包含：不修改项目任何现有代码，只在运行时重定向 ffmpeg 路径与导出目录。

非交互场景（后台挂机）也能用：答案按行喂给 stdin，EOF 自动取默认值
    printf '/root/autodl-tmp/voice\\n\\n\\n' | nohup python autodl_train.py > run.log 2>&1 &
数据集路径也可直接作为第一个参数传入（其余仍走问答）：
    python autodl_train.py /root/autodl-tmp/voice
"""
import importlib.util
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

# ── 0. 控制台编码 + 工作目录 ──────────────────────────────────────────
# 项目内大量路径是相对 cwd 的（assets/hubert、assets/rmvpe、assets/configs），
# 统一切到脚本所在目录（=项目根）后全部自然生效，无需改动任何代码。
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 云上系统盘只有 30G，训练产物一律走数据盘；本地（无 /root/autodl-tmp）降级到项目内
_DEFAULT_CLOUD_ROOT = Path("/root/autodl-tmp")
if os.environ.get("RVC_OUT_ROOT", "").strip():
    DATA_ROOT = Path(os.environ["RVC_OUT_ROOT"].strip()).expanduser()
elif _DEFAULT_CLOUD_ROOT.is_dir():
    DATA_ROOT = _DEFAULT_CLOUD_ROOT
else:
    DATA_ROOT = PROJECT_ROOT / "autodl-tmp"
LOGS_ROOT = DATA_ROOT / "logs"
MODELS_ROOT = DATA_ROOT / "models"

# 环境检测用（纯元数据查询，不真正执行模块）
REQUIRED_PACKAGES = [
    ("torch", "torch", "PyTorch 深度学习框架（训练核心）"),
    ("numpy", "numpy", "数组计算"),
    ("scipy", "scipy", "滤波器 / 重采样"),
    ("librosa", "librosa", "音频加载与切片 RMS"),
    ("soundfile", "soundfile", "wav 读写（预处理产物）"),
    ("transformers", "transformers", "HuBERT 特征提取"),
]
# 训练用不到，缺了不报错（本地 GUI 才需要）
OPTIONAL_PACKAGES = [
    ("PySide6", "PySide6", "GUI（云端不需要）"),
    ("sounddevice", "sounddevice", "实时音频 IO（云端不需要）"),
    ("faiss", "faiss", "推理侧 index 检索（云端训练不需要）"),
]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
# libsndfile 原生支持的格式；其余一律走 ffmpeg 解码
NATIVE_EXTS = {".wav", ".flac", ".ogg"}

EXIT_OK, EXIT_ENV, EXIT_RUNTIME = 0, 1, 2

BAR = "─" * 66

DEFAULTS = {
    "exp": "test",
    "sr": "48k",
    "epochs": 100,
    "lr": 1e-4,
    "save_every": 20,
    "keep_ckpts": 1,
    "per": 3.7,
    "keep_models": 0,  # 导出模型保留数 0 = 全部保留（不淘汰，固定，不再暴露给用户）
}


class Cancelled(Exception):
    """用户 Ctrl+C 取消。"""


class EnvFatal(Exception):
    """环境体检致命项缺失。"""


# ── 1. 日志 ──────────────────────────────────────────────────────────
class TrainLogger:
    """控制台 + 日志文件。

    启动即写到引导日志（输出根/logs/autodl_train.log），实验名确定后 reattach 到
    <实验目录>/autodl_train.log，并把引导阶段的内容原样搬到新文件头部，最终只留一份。
    """

    def __init__(self):
        self.log_file = None
        self._fp = None
        self._buffer = []
        self._t0 = time.time()

    def attach(self, log_file: Path):
        self.log_file = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_file.open("a", encoding="utf-8")
        for level, msg in self._buffer:
            self._write(level, msg)
        self._buffer.clear()

    def reattach(self, log_file: Path):
        """把日志搬到新位置，旧内容搬到新文件头部。"""
        old_file, old_fp = self.log_file, self._fp
        carried = []
        try:
            if old_fp:
                old_fp.close()
            if old_file and old_file.exists() and old_file != log_file:
                carried = old_file.read_text(encoding="utf-8").splitlines()
                old_file.unlink()
        except Exception:
            pass
        self._buffer.clear()
        self.attach(log_file)
        if carried:
            self._fp.write("—— 以下为实验名确定前的体检记录 ——\n")
            for line in carried:
                self._fp.write(line + "\n")
            self._fp.write("——————————————————\n")
            self._fp.flush()

    @staticmethod
    def _stamp():
        return time.strftime("%m-%d %H:%M:%S")

    def _write(self, level, msg):
        self._fp.write(f"[{self._stamp()}] [{level:<5}] {msg}\n")
        self._fp.flush()

    def log(self, msg: str, level: str = "INFO"):
        line = f"[{self._stamp()}] [{level:<5}] {msg}" if self._fp else msg
        print(line, flush=True)
        if self._fp:
            self._write(level, msg)
        else:
            self._buffer.append((level, msg))

    def file_only(self, msg: str, level: str = "INFO"):
        """只写日志、不上屏（Trainer 自带的 epoch loss 行与下面的 EPOCH 行重复）。"""
        if self._fp:
            self._write(level, msg)
        else:
            self._buffer.append((level, msg))

    def plain(self, msg: str = ""):
        """只上屏、不进日志的分隔/提示行。"""
        print(msg, flush=True)

    def section(self, title: str):
        self.plain(BAR)
        self.log(f"【{title}】")
        self.plain(BAR)

    def elapsed(self):
        return time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - self._t0))

    def close(self):
        try:
            if self._fp:
                self._fp.close()
        except Exception:
            pass


# ── 2. 交互提问 ──────────────────────────────────────────────────────
def _clean(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def ask(prompt: str, default="", cast=None, check=None, allow_blank=False):
    """提问。default 为空串 = 必填；EOF（管道结束）自动取默认值。"""
    while True:
        hint = f"（默认 {default}）" if default != "" else ""
        try:
            raw = _clean(input(f"{prompt}{hint}: "))
        except EOFError:
            print(flush=True)
            if allow_blank or default != "":
                raw = ""
            else:
                # 输入已结束还缺必填项，再问也是死循环
                raise EnvFatal(
                    f"必填项未填写，且输入已结束: {prompt}\n"
                    "  提示：非交互运行时把答案按行喂给 stdin，或把数据集路径作为第一个参数传入"
                ) from None
        except KeyboardInterrupt:
            print(flush=True)
            raise Cancelled() from None
        if raw == "":
            if allow_blank:
                return ""
            if default != "":
                raw = str(default)
            else:
                print("  × 该项必填，请重新输入", flush=True)
                continue
        if cast is not None:
            try:
                val = cast(raw)
            except Exception:
                print(f"  × 格式不对：{raw}", flush=True)
                continue
        else:
            val = raw
        if check is not None:
            ok, msg = check(val)
            if not ok:
                print(f"  × {msg}", flush=True)
                continue
        return val


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = _clean(input(f"{prompt} [{hint}]: ")).lower()
        except EOFError:
            print(flush=True)
            return default
        except KeyboardInterrupt:
            print(flush=True)
            raise Cancelled() from None
        if raw == "":
            return default
        if raw in ("y", "yes", "是", "1"):
            return True
        if raw in ("n", "no", "否", "0"):
            return False
        print("  × 请输入 y 或 n", flush=True)


def _as_int(raw):
    return int(str(raw).strip())


def _as_float(raw):
    return float(str(raw).strip())


def _as_sr(raw):
    """接受 48k / 48 / 48000 三种写法，统一返回 '48k'。"""
    text = str(raw).strip().lower().replace("hz", "")
    if text.endswith("k"):
        return f"{_as_int(text[:-1])}k"
    value = _as_int(text)
    return f"{value // 1000}k" if value >= 1000 else f"{value}k"


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024


def _human_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ── 3. 环境体检 ──────────────────────────────────────────────────────
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
    """返回 (路径, 来源)。非 Windows 平台跳过项目内的 Windows exe。

    项目内 assets/ffmpeg/ffmpeg.exe 是 Windows 二进制：Linux/macOS 上
    没有执行权限位，直接执行会 PermissionError（云上已踩过）。
    """
    local_exe = PROJECT_ROOT / "assets" / "ffmpeg" / "ffmpeg.exe"
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
    """定位 ffmpeg 并实际验证可执行；找到就重定向 loader 的硬编码路径。

    rvc/audio/loader.py 写死了 assets/ffmpeg/ffmpeg.exe（Windows 专用二进制），
    云端必须改指向系统 ffmpeg，否则 mp3/m4a 等格式无法解码。
    这里只改模块变量，不碰源文件。
    """
    local_exe = PROJECT_ROOT / "assets" / "ffmpeg" / "ffmpeg.exe"
    chosen, source = _locate_ffmpeg()
    if not chosen:
        log.log("ffmpeg      : 未找到（系统 PATH / assets/ffmpeg 都没有）", "WARN")
        log.log("  素材若含 mp3/m4a/aac 会直接失败，安装：apt install -y ffmpeg", "WARN")
        # 防御：把 loader 硬编码的 Windows exe 指空，让它报清晰的 FileNotFoundError
        # 而不是在 Linux 上执行 .exe 报 PermissionError
        try:
            import rvc.audio.loader as _loader

            _loader._FFMPEG = Path("")
        except Exception:
            pass
        return ""
    # 实际验证可执行（体检阶段就暴露问题，而不是训练中途才炸）
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
    if Path(chosen).resolve() != local_exe.resolve():
        try:
            import rvc.audio.loader as _loader

            _loader._FFMPEG = Path(chosen)  # 函数体内是全局查找，改模块属性即生效
            log.log("             → 已重定向 rvc.audio.loader 的 ffmpeg 路径")
        except Exception as exc:
            log.log(f"重定向 ffmpeg 路径失败（忽略）: {exc}", "WARN")
    return chosen


def _require_ffmpeg(log: TrainLogger, files: list[Path]):
    """素材里有非原生格式却没有 ffmpeg → 直接失败（否则训练中途才炸）。"""
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
    assets = PROJECT_ROOT / "assets"

    for sr_k in (40, 48):
        cfg = assets / "configs" / f"{sr_k}ktrain_config.json"
        if not cfg.exists():
            fatal.append(f"训练配置缺失: assets/configs/{sr_k}ktrain_config.json（应随代码仓库一起上传）")

    hubert_dir = assets / "hubert"
    for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        path = hubert_dir / f
        if not path.exists():
            fatal.append(f"HuBERT 权重缺失: assets/hubert/{f}")
        else:
            log.log(f"  HuBERT {f:<24} {_human_size(path.stat().st_size)}")

    rmvpe = assets / "rmvpe" / "rmvpe.pt"
    if not rmvpe.exists():
        fatal.append("RMVPE 权重缺失: assets/rmvpe/rmvpe.pt")
    else:
        log.log(f"  RMVPE  rmvpe.pt                {_human_size(rmvpe.stat().st_size)}")

    ready = {}
    for sr_k in (40, 48):
        ok = True
        for key in ("G", "D"):
            path = assets / "pretrained" / f"f0{key}{sr_k}k.pth"
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
        log.log("  · assets/hubert/   （config.json + preprocessor_config.json + pytorch_model.bin）", "ERROR")
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
    if not input_dir.is_dir():
        return []
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _probe_dataset(log: TrainLogger, input_dir: Path, files: list[Path]):
    """统计素材：文件数、体积、总时长、采样率分布（超过 300 个只抽样后按比例外推）。"""
    total_bytes = sum(p.stat().st_size for p in files)
    log.log(f"素材目录    : {input_dir.resolve()}")
    log.log(f"音频文件    : {len(files)} 个，共 {_human_size(total_bytes)}")

    import soundfile as sf

    limit = 300
    sample = files[:limit]
    seconds, sr_dist, scanned = 0.0, {}, 0
    for path in sample:
        try:
            info = sf.info(str(path))
            seconds += info.frames / info.samplerate
            sr_dist[info.samplerate] = sr_dist.get(info.samplerate, 0) + 1
            scanned += 1
        except Exception:
            continue
    if scanned:
        if scanned < len(files):
            seconds = seconds * len(files) / scanned
        log.log(f"总时长      : 约 {_human_dur(seconds)}" + ("（抽样估算）" if scanned < len(files) else ""))
        log.log("采样率分布  : " + ", ".join(f"{k}Hz×{v}" for k, v in sorted(sr_dist.items())))
    if len(files) < 10:
        log.log("素材文件偏少（<10 个），音质与稳定性会明显下降，建议 30 分钟以上干净人声", "WARN")
    if scanned and seconds < 300:
        log.log("总时长不足 5 分钟，容易过拟合，建议 30 分钟以上", "WARN")


# ── 4. 参数向导 ──────────────────────────────────────────────────────
def _detect_ckpt_epoch(exp_dir: Path) -> int:
    from rvc.train.ckpt_utils import checkpoints_dir as _ckpt_dir
    from rvc.train.ckpt_utils import checkpoint_epoch, latest_checkpoint_path

    directory = _ckpt_dir(exp_dir)
    g = latest_checkpoint_path(str(directory), "G") or latest_checkpoint_path(str(exp_dir), "G")
    d = latest_checkpoint_path(str(directory), "D") or latest_checkpoint_path(str(exp_dir), "D")
    if not g or not d:
        return 0
    return min(checkpoint_epoch(g), checkpoint_epoch(d))


def run_wizard(log: TrainLogger, device: str, fp16: bool, vram_gb: float, argv_dir: str) -> dict:
    cfg = {"device": device, "fp16": fp16}

    log.section("第 3 步 / 训练参数（直接回车 = 使用默认值）")

    # 1) 数据集路径
    def _dir_ok(text):
        path = Path(text).expanduser()
        if not path.is_dir():
            return False, f"目录不存在: {path}"
        if not _scan_audio(path):
            return False, f"目录里没有音频文件（支持 {', '.join(sorted(AUDIO_EXTS))}）: {path}"
        return True, ""

    input_dir = Path(ask("数据集路径", argv_dir, check=_dir_ok)).expanduser()
    files = _scan_audio(input_dir)
    _probe_dataset(log, input_dir, files)
    _require_ffmpeg(log, files)

    # 2) 实验名（决定产物落点，先拿到才好把日志搬过去）
    exp_name = ask("实验名", DEFAULTS["exp"])
    exp_dir = LOGS_ROOT / exp_name
    model_dir = MODELS_ROOT / exp_name
    log.reattach(exp_dir / "autodl_train.log")
    log.log(f"实验目录    : {exp_dir}")
    log.log(f"模型目录    : {model_dir}")
    log.log(f"日志文件    : {log.log_file}")

    # 3) 续训检测（不提问：检测到就续训；要重来请手动删掉 checkpoint 目录）
    ckpt_epoch = _detect_ckpt_epoch(exp_dir)
    if ckpt_epoch:
        log.log(f"检测到已有 checkpoint（已训练到 epoch {ckpt_epoch}）→ 自动续训")
        log.log(f"  要重新训练请先删掉: {exp_dir / '4_checkpoints'}")
    elif exp_dir.exists():
        log.log("实验目录已存在但无 checkpoint，将复用其中已有的切片/特征")

    # 4) 采样率
    def _sr_ok(value):
        if value not in ("40k", "48k"):
            return False, "只能填 40k 或 48k"
        return True, ""

    sr = ask("采样率 40k / 48k", DEFAULTS["sr"], cast=_as_sr, check=_sr_ok)
    sr_hz = 48000 if sr == "48k" else 40000
    if sr == "40k":
        log.log("40k：显存占用更小、训练更快，高频细节略少于 48k", "WARN")

    # 5) batch size
    batch = ask("batch size", str(_suggest_batch(vram_gb, sr_hz)), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    # 6) 学习率
    lr = ask(
        "学习率（1e-4=0.0001；太大易炸，太小收敛慢）",
        f"{DEFAULTS['lr']:g}",
        cast=_as_float,
        check=lambda v: (0 < v <= 0.1, "需要在 0 ~ 0.1 之间"),
    )
    if lr > 5e-4:
        log.log("学习率偏大，GAN 容易崩（loss 变 nan）；新手建议 1e-4 附近", "WARN")
    elif lr < 1e-5:
        log.log("学习率偏小，100 轮大概率训不出东西", "WARN")

    # 7) 总轮次
    def _epochs_ok(value):
        if value < 1:
            return False, "至少 1 轮"
        if ckpt_epoch and value <= ckpt_epoch:
            return False, f"已训到 epoch {ckpt_epoch}，总轮次必须大于它（填的是目标终点，不是新增轮数）"
        return True, ""

    epochs = ask("训练轮次", str(DEFAULTS["epochs"]), cast=_as_int, check=_epochs_ok)

    # 8) 保存间隔
    save_every = ask("每多少轮保存一次", str(DEFAULTS["save_every"]), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    # 9) 保留 checkpoint 组数（G+D 各一组约 0.8GB，只留最新几组即可）
    keep_ckpts = ask("保留最近几组 checkpoint", str(DEFAULTS["keep_ckpts"]), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    sr_k = sr_hz // 1000
    pretrain_g = PROJECT_ROOT / "assets" / "pretrained" / f"f0G{sr_k}k.pth"
    pretrain_d = PROJECT_ROOT / "assets" / "pretrained" / f"f0D{sr_k}k.pth"

    cfg.update(
        input_dir=str(input_dir),
        exp_name=exp_name,
        exp_dir=exp_dir,
        model_dir=model_dir,
        sr=sr,
        sr_hz=sr_hz,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        save_every=save_every,
        per=DEFAULTS["per"],
        keep_models=DEFAULTS["keep_models"],
        keep_ckpts=keep_ckpts,
        ckpt_epoch=ckpt_epoch,
        pretrain_g=str(pretrain_g) if pretrain_g.exists() else "",
        pretrain_d=str(pretrain_d) if pretrain_d.exists() else "",
    )
    return cfg


def print_summary(log: TrainLogger, cfg: dict):
    log.section("配置确认")
    ckpt = cfg["ckpt_epoch"]
    rows = [
        ("数据集", cfg["input_dir"]),
        ("实验名", cfg["exp_name"]),
        ("实验目录", str(cfg["exp_dir"])),
        ("模型目录", str(cfg["model_dir"])),
        ("采样率", cfg["sr"]),
        ("总轮次", str(cfg["epochs"]) + (f"（从 epoch {ckpt + 1} 续训）" if ckpt else "")),
        ("batch size", cfg["batch_size"]),
        ("学习率", f"{cfg['lr']:g}"),
        ("保存间隔", f"每 {cfg['save_every']} 轮"),
        ("切片时长", f"{cfg['per']}s（固定）"),
        ("保留 checkpoint", f"最近 {cfg['keep_ckpts']} 组"),
        ("设备", f"{cfg['device']}  fp16={cfg['fp16']}"),
        ("底模", cfg["pretrain_g"] or "无（从零训练）"),
    ]
    for key, value in rows:
        log.plain(f"  {key:<14} {value}")
        log.log(f"配置 {key}: {value}")
    log.plain(BAR)


# ── 5. 训练步骤 ──────────────────────────────────────────────────────
class _StopFlag:
    requested = False
    trainer = None


STOP = _StopFlag()


def _install_signal_handlers(log: TrainLogger):
    def handler(signum, _frame):
        if STOP.requested:
            log.log("再次收到终止信号，强制退出", "WARN")
            sys.exit(EXIT_RUNTIME)
        STOP.requested = True
        name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else f"SIGTERM ({signum})"
        log.log(f"收到 {name} → 当前 epoch 跑完会保存后退出", "WARN")
        if STOP.trainer is not None:
            STOP.trainer.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def _make_progress(log: TrainLogger, label: str):
    state = {"last": -1, "t0": time.time()}

    def cb(done, total):
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct != state["last"] and (pct % 5 == 0 or done == total):
            state["last"] = pct
            speed = done / max(time.time() - state["t0"], 1e-6)
            left = (total - done) / speed if speed > 0 else 0
            log.log(f"{label}: {done}/{total} ({pct}%)  速度 {speed:.1f} 条/s  剩余 {_human_dur(left)}", "进度")

    return cb


def step_preprocess(log: TrainLogger, cfg: dict):
    from rvc.train.preprocess import PreProcessor

    log.section("步骤 1/4 预处理（切片 + 重采样）")
    exp_dir = cfg["exp_dir"]
    log.log(f"切片时长 {cfg['per']}s，目标采样率 {cfg['sr_hz']} Hz")
    t0 = time.time()
    processor = PreProcessor(cfg["input_dir"], str(exp_dir), cfg["sr_hz"], per=cfg["per"])
    count = processor.run(_make_progress(log, "预处理"))
    secs = time.time() - t0
    log.log(f"预处理完成：{count} 个源文件，用时 {_human_dur(secs)}")

    gt = sorted((exp_dir / "0_gt_wavs").glob("*.wav"))
    if gt:
        bytes_total = sum(p.stat().st_size for p in gt)
        log.log(f"生成切片：{len(gt)} 条，约 {_human_dur(bytes_total / 2 / cfg['sr_hz'])} 音频，{_human_size(bytes_total)}")
    if len(gt) < 50:
        log.log("切片数偏少（<50），训练容易过拟合，建议补充素材", "WARN")


def step_f0(log: TrainLogger, cfg: dict):
    from rvc.train.extract_f0 import TrainF0Extractor

    log.section("步骤 2/4 提取 F0（RMVPE）")
    t0 = time.time()
    extractor = TrainF0Extractor(cfg["device"], cfg["fp16"])
    n = extractor.run(str(cfg["exp_dir"]), _make_progress(log, "F0"), stop_check=lambda: STOP.requested)
    secs = time.time() - t0
    log.log(f"F0 提取完成：{n} 条，用时 {_human_dur(secs)}（{n / max(secs, 1e-6):.1f} 条/s）")


def step_feature(log: TrainLogger, cfg: dict):
    from rvc.train.extract_feature import HuBERTExtractor

    log.section("步骤 3/4 提取 HuBERT 特征")
    t0 = time.time()
    extractor = HuBERTExtractor(cfg["device"], cfg["fp16"])
    n = extractor.run(str(cfg["exp_dir"]), _make_progress(log, "特征"), stop_check=lambda: STOP.requested)
    secs = time.time() - t0
    log.log(f"特征提取完成：{n} 条，用时 {_human_dur(secs)}（{n / max(secs, 1e-6):.1f} 条/s）")


def _features_ready(exp_dir: Path) -> bool:
    gt = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
    if gt == 0:
        return False
    f0 = len(list((exp_dir / "2a_f0").glob("*.npy")))
    f0nsf = len(list((exp_dir / "2b-f0nsf").glob("*.npy")))
    feat = len(list((exp_dir / "3_feature768").glob("*.npy")))
    return f0 >= gt and f0nsf >= gt and feat >= gt


def _gpu_mem() -> tuple[float, float]:
    """(当前分配 GB, 峰值 GB)"""
    try:
        import torch

        return torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        return 0.0, 0.0


def step_train(log: TrainLogger, cfg: dict):
    from rvc.train.preprocess import generate_filelist
    from rvc.train.trainer import TrainConfig, Trainer
    import rvc.train.trainer as trainer_mod

    log.section("步骤 4/4 训练")
    exp_dir = cfg["exp_dir"]
    model_dir = cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    # 导出目录重定向到数据盘（trainer 内 WEIGHTS_DIR 是模块级常量，改它即可，源文件不动）
    trainer_mod.WEIGHTS_DIR = model_dir
    log.log(f"导出模型目录已重定向到: {model_dir}")

    # 从头训练时清掉同名旧模型：epoch 号会重新从 1 开始，
    # 而淘汰是按 epoch 数值排的，不清理的话旧的 e5 会一直压住新训的 e1/e2
    if _detect_ckpt_epoch(exp_dir) == 0:
        stale = sorted(model_dir.glob(f"{exp_dir.name}_e*.pth"))
        if stale:
            for path in stale:
                path.unlink()
            log.log(f"本次从头训练，已删除旧的导出模型 {len(stale)} 个（epoch 号会重新计数）", "WARN")

    _, count = generate_filelist(str(exp_dir))
    log.log(f"训练样本数  : {count}")
    if count == 0:
        raise RuntimeError(f"没有可训练样本（特征或 F0 缺失），请删除 {exp_dir} 后重跑")

    for name in ("pretrain_g", "pretrain_d"):
        path = cfg[name]
        if path and not Path(path).exists():
            raise RuntimeError(f"预训练模型不存在: {path}")

    train_config = TrainConfig(
        exp_dir=str(exp_dir),
        sr=cfg["sr_hz"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        save_every_epoch=cfg["save_every"],
        learning_rate=cfg["lr"],
        pretrain_g=cfg["pretrain_g"],
        pretrain_d=cfg["pretrain_d"],
        fp16_run=cfg["fp16"],
        device=cfg["device"],
        keep_ckpts=cfg["keep_ckpts"],
        keep_models=cfg["keep_models"],
    )

    state = {"t_epoch": time.time(), "sum": None, "count": 0, "batches": 0, "samples": 0, "last_ckpt": 0, "ema_secs": None}

    def _reset_epoch():
        state["sum"] = {"d": 0.0, "g": 0.0, "mel": 0.0, "kl": 0.0, "fm": 0.0}
        state["count"] = 0
        state["t_epoch"] = time.time()

    _reset_epoch()

    def on_batch(epoch, batch, total):
        state["batches"] += 1

    def on_loss(info):
        # 每个 epoch 的第一个 batch 重置统计与计时（首个 epoch 的 setup 时间不该算进速度）
        if info["batch"] == 1:
            _reset_epoch()
        s = state["sum"]
        for key, field in (("d", "loss_d"), ("g", "loss_g"), ("mel", "loss_mel"), ("kl", "loss_kl"), ("fm", "loss_fm")):
            s[key] += info[field]
        state["count"] += 1

    def on_epoch(epoch, total):
        s, n = state["sum"], max(state["count"], 1)
        secs = time.time() - state["t_epoch"]
        # ETA 用 EMA 平滑耗时，避免单轮波动（8.6→9.5→8.1s）让剩余时间来回跳
        ema = secs if state["ema_secs"] is None else state["ema_secs"] * 0.7 + secs * 0.3
        state["ema_secs"] = ema
        left = ema * (total - epoch)
        eta = _human_dur(left) if left > 0 else "--"
        lr_now = ""
        if STOP.trainer is not None and hasattr(STOP.trainer, "optim_g") and STOP.trainer.optim_g is not None:
            lr_now = f"lr {STOP.trainer.optim_g.param_groups[0]['lr']:.2e} | "
        cur, peak = _gpu_mem()
        mem = f" | 显存 现{cur:.1f}/峰{peak:.1f}GB" if cfg["device"].startswith("cuda") else ""
        speed = f"{state['count'] / max(secs, 1e-6):.1f} it/s"
        log.log(
            f"epoch {epoch:>4}/{total} | {lr_now}"
            f"D {s['d'] / n:.4f} G {s['g'] / n:.4f} Mel {s['mel'] / n:.4f} "
            f"KL {s['kl'] / n:.4f} FM {s['fm'] / n:.4f} | {secs:.1f}s {speed} 剩余 {eta}{mem}",
            "EPOCH",
        )
        _reset_epoch()

    def on_trainer_log(msg: str):
        # Trainer 每轮会自己往 train.log 落一行 epoch loss，与上面的 EPOCH 行内容重复；
        # 这里只留档不上屏，控制台保持单行可读
        if msg.startswith("epoch "):
            log.file_only(msg)
        else:
            log.log(msg)

    trainer = Trainer(train_config, on_epoch, on_trainer_log, on_loss, on_batch)
    STOP.trainer = trainer
    t0 = time.time()
    try:
        output = trainer.train()
    finally:
        trainer.cleanup()
        STOP.trainer = None

    log.log(f"训练结束，总用时 {_human_dur(time.time() - t0)}，共 {state['batches']} 个 batch")
    if cfg["device"].startswith("cuda"):
        log.log(f"显存峰值    : {_gpu_mem()[1]:.2f} GB")
    log.log(f"最终模型    : {output}")
    return output


# ── 6. 主流程 ────────────────────────────────────────────────────────
def main() -> int:
    # Trainer 里 scheduler.step() 先于 optimizer.step() 的既有写法会每轮刷两条警告，
    # 属于原代码行为，这里只在脚本侧静音，不改源文件
    warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")

    argv_dir = ""
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            argv_dir = arg

    log = TrainLogger()
    try:
        log.attach(DATA_ROOT / "logs" / "autodl_train.log")
        log.section("RVC 云 GPU 训练向导")

        log.section("第 1 步 / 环境检查")
        _probe_platform(log)
        _probe_disk(log)
        if _probe_packages(log):
            return EXIT_ENV
        device, fp16, vram_gb = _probe_gpu(log)
        _probe_ffmpeg(log)
        log.log(f"最终设备    : device={device}  fp16={fp16}")

        log.section("第 2 步 / 预训练模型检查")
        _probe_assets(log)

        cfg = run_wizard(log, device, fp16, vram_gb, argv_dir)
        print_summary(log, cfg)

        if not ask_yes("开始训练？（训练中 Ctrl+C = 本轮跑完保存后退出）", default=True):
            log.log("已取消，未开始训练", "WARN")
            return EXIT_OK

        _install_signal_handlers(log)
        exp_dir = cfg["exp_dir"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        from rvc.train.preprocess import manifest_diff_reason

        reason = manifest_diff_reason(exp_dir, cfg["input_dir"], cfg["sr_hz"], cfg["per"])
        if reason:
            log.log(f"需要重新预处理：{reason}")
            step_preprocess(log, cfg)
        else:
            log.log(f"切片与素材一致，跳过预处理（要重建请删 {exp_dir / 'manifest.json'}）")

        if not STOP.requested:
            if _features_ready(exp_dir):
                log.log("F0 与 HuBERT 特征已完整，跳过提取")
            else:
                step_f0(log, cfg)
                if not STOP.requested:
                    step_feature(log, cfg)

        if not STOP.requested:
            step_train(log, cfg)

        log.section("全部完成")
        log.log(f"总耗时      : {log.elapsed()}")
        models = sorted(cfg["model_dir"].glob("*.pth")) if cfg["model_dir"].is_dir() else []
        if models:
            log.log(f"导出模型目录: {cfg['model_dir']}")
            for path in models:
                log.log(f"  · {path.name}  {_human_size(path.stat().st_size)}")
        log.log(f"checkpoint  : {exp_dir / '4_checkpoints'}")
        log.log(f"训练日志    : {exp_dir / 'train.log'}")
        log.log(f"向导日志    : {log.log_file}")
        return EXIT_OK

    except Cancelled:
        log.log("已取消", "WARN")
        return EXIT_OK
    except EnvFatal as exc:
        for line in str(exc).splitlines():
            log.log(line, "ERROR")
        log.log("环境检查未通过，已中止（不会开始训练）", "ERROR")
        return EXIT_ENV
    except KeyboardInterrupt:
        log.log("已中断", "WARN")
        return EXIT_RUNTIME
    except Exception as exc:
        log.log(f"运行出错: {type(exc).__name__}: {exc}", "ERROR")
        import traceback

        for line in traceback.format_exc().splitlines():
            log.log(line, "ERROR")
        log.log("常见问题：显存不足（OOM）请调小 batch size；素材格式问题请确认 ffmpeg 已装", "ERROR")
        return EXIT_RUNTIME
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
