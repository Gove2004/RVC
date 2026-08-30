"""RVC 云 GPU（AutoDL / 恒源云 / 揽睿星舟等）训练向导。

交互式，无需记任何参数：
    python autodl_train.py

流程：环境体检 → 逐项提问（全部带默认值，直接回车即可）→ 打印配置摘要 → 开始训练。
自包含：不修改项目任何现有代码，只在运行时把 ffmpeg 路径重定向到系统 ffmpeg。

非交互场景（后台/脚本）也能用：答案从 stdin 逐行读取，EOF 自动取默认值
    printf '/root/autodl-tmp/voice\\n\\n\\n' | nohup python autodl_train.py > run.log 2>&1 &
数据集路径也可直接作为第一个参数传入（其余仍走问答）：
    python autodl_train.py /root/autodl-tmp/voice
"""
import importlib.util
import os
import shutil
import signal
import sys
import time
import warnings
from pathlib import Path

# ── 0. 控制台编码 + 工作目录 ──────────────────────────────────────────
# 项目内大量路径是相对 cwd 的（assets/hubert、assets/rmvpe、logs、assets/configs），
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

BAR = "─" * 62


class Cancelled(Exception):
    """用户 Ctrl+C 取消。"""


class EnvFatal(Exception):
    """环境体检致命项缺失。"""


# ── 1. 日志 ──────────────────────────────────────────────────────────
class TrainLogger:
    """控制台 + 日志文件。日志路径在实验名确定后才 attach，之前的行先缓冲。"""

    def __init__(self):
        self.log_file = None
        self._fp = None
        self._buffer = []
        self._t0 = time.time()

    def attach(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.log_file.open("a", encoding="utf-8")
        for level, msg in self._buffer:
            self._write(level, msg)
        self._buffer.clear()

    @staticmethod
    def _stamp():
        return time.strftime("%H:%M:%S")

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

    def plain(self, msg: str = ""):
        """只上屏、不进日志的分隔/提示行。"""
        print(msg, flush=True)

    def section(self, title: str):
        self.plain(BAR)
        self.log(f"  {title}")
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


# ── 3. 环境体检 ──────────────────────────────────────────────────────
def _probe_packages(log: TrainLogger) -> list[str]:
    missing = []
    for import_name, pip_name, desc in REQUIRED_PACKAGES:
        if importlib.util.find_spec(import_name) is None:
            missing.append((pip_name, desc))
    if missing:
        log.log("缺少以下必需依赖：", "ERROR")
        for pip_name, desc in missing:
            log.log(f"  · {pip_name:<14} {desc}", "ERROR")
        log.plain()
        log.log("请先安装（AutoDL 镜像一般已带 torch，只补缺的即可）：", "ERROR")
        log.log(f"  pip install {' '.join(p for p, _ in missing)}", "ERROR")
        log.plain()
        log.log("注意：torch 必须与镜像 CUDA 版本匹配，重装前先确认", "ERROR")
        log.log('  python -c "import torch; print(torch.__version__, torch.version.cuda)"', "ERROR")
    else:
        missing_optional = [pip for imp, pip, _ in OPTIONAL_PACKAGES if importlib.util.find_spec(imp) is None]
        log.log("必需依赖齐全" + (f"（云端不需要的没装: {', '.join(missing_optional)}）" if missing_optional else ""))
    return [p for p, _ in missing]


def _probe_gpu(log: TrainLogger) -> tuple[str, bool, float]:
    """返回 (device, use_fp16, 显存GB)。"""
    import torch

    log.log(f"PyTorch      : {torch.__version__}")
    log.log(f"CUDA 编译版本: {torch.version.cuda or '无（CPU 版 torch）'}")

    if not torch.cuda.is_available():
        log.log("未检测到可用 CUDA GPU —— 训练会退化到 CPU，速度极慢（不推荐）", "WARN")
        log.log("  检查：nvidia-smi 是否能看到显卡；torch 是否为 CUDA 版", "WARN")
        return "cpu", False, 0.0

    idx = 0
    name = torch.cuda.get_device_name(idx)
    total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    cc = torch.cuda.get_device_capability(idx)
    log.log(f"GPU          : {idx} {name}")
    log.log(f"显存         : {total:.1f} GB")
    log.log(f"算力         : sm_{cc[0]}{cc[1]}")

    use_fp16 = True
    if cc[0] < 7:
        log.log("算力 < 7.0（如 P40 / 1080Ti），fp16 收益有限且可能不稳定 → 自动改用 fp32", "WARN")
        use_fp16 = False
    return f"cuda:{idx}", use_fp16, total


def _scan_audio(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _probe_ffmpeg(log: TrainLogger, files: list[Path]) -> None:
    """定位 ffmpeg；找到系统 ffmpeg 就运行时重定向 loader 的硬编码路径。

    rvc/audio/loader.py 写死了 assets/ffmpeg/ffmpeg.exe（Windows 专用二进制），
    云端必须改指向系统 ffmpeg，否则 mp3/m4a 等格式无法解码。
    这里只改模块变量，不碰源文件。
    """
    local_exe = PROJECT_ROOT / "assets" / "ffmpeg" / "ffmpeg.exe"
    system_ffmpeg = shutil.which("ffmpeg")
    env_ffmpeg = os.environ.get("RVC_FFMPEG", "").strip()

    if env_ffmpeg and Path(env_ffmpeg).exists():
        chosen, source = env_ffmpeg, "环境变量 RVC_FFMPEG"
    elif system_ffmpeg:
        chosen, source = system_ffmpeg, "系统 PATH"
    elif local_exe.exists():
        chosen, source = str(local_exe), "项目内 assets/ffmpeg/ffmpeg.exe"
    else:
        chosen, source = "", ""

    foreign = [p for p in files if p.suffix.lower() not in NATIVE_EXTS]
    if chosen:
        log.log(f"ffmpeg       : {chosen}（来自 {source}）")
        if Path(chosen).resolve() != local_exe.resolve():
            try:
                import rvc.audio.loader as _loader

                _loader._FFMPEG = Path(chosen)  # 函数体内是全局查找，改模块属性即生效
                log.log("             → 已重定向 rvc.audio.loader 的 ffmpeg 路径")
            except Exception as exc:
                log.log(f"重定向 ffmpeg 路径失败（忽略）: {exc}", "WARN")
        return

    log.log("未找到 ffmpeg（系统 PATH / assets/ffmpeg 都没有）", "WARN")
    if foreign:
        raise EnvFatal(
            f"素材里有 {len(foreign)} 个非 wav/flac/ogg 文件（如 {foreign[0].name}），缺少 ffmpeg 会直接失败。\n"
            "  安装：apt install -y ffmpeg   （或 conda install -c conda-forge ffmpeg）"
        )
    log.log("素材全是 wav/flac/ogg，libsndfile 可直接读取 —— 不影响训练", "WARN")
    log.log("  后续若加入 mp3/m4a，先装 ffmpeg：apt install -y ffmpeg", "WARN")


def _probe_assets(log: TrainLogger, sr_hz: int) -> None:
    sr_k = sr_hz // 1000
    fatal, warns = [], []

    cfg = PROJECT_ROOT / "assets" / "configs" / f"{sr_k}ktrain_config.json"
    if not cfg.exists():
        fatal.append(f"训练配置缺失: assets/configs/{sr_k}ktrain_config.json（应随代码仓库一起上传）")

    hubert_dir = PROJECT_ROOT / "assets" / "hubert"
    for f in ("config.json", "pytorch_model.bin"):
        if not (hubert_dir / f).exists():
            fatal.append(f"HuBERT 权重缺失: assets/hubert/{f}")

    rmvpe = PROJECT_ROOT / "assets" / "rmvpe" / "rmvpe.pt"
    if not rmvpe.exists():
        fatal.append("RMVPE 权重缺失: assets/rmvpe/rmvpe.pt")

    for key in ("G", "D"):
        p = PROJECT_ROOT / "assets" / "pretrained" / f"f0{key}{sr_k}k.pth"
        if not p.exists():
            warns.append(f"预训练底模缺失: assets/pretrained/f0{key}{sr_k}k.pth（将从零训练，收敛明显变慢）")

    if fatal:
        for msg in fatal:
            log.log(msg, "ERROR")
        log.plain()
        log.log("以上权重都在 .gitignore 里，git clone 不会带下来，需要单独上传：", "ERROR")
        log.log("  · assets/hubert/   （config.json + pytorch_model.bin + preprocessor_config.json）", "ERROR")
        log.log("  · assets/rmvpe/rmvpe.pt", "ERROR")
        log.log("  上传方式：AutoDL 网盘 / scp / rsync，放到项目根对应目录", "ERROR")
        raise EnvFatal("模型权重缺失")

    for msg in warns:
        log.log(msg, "WARN")
    log.log("必需权重齐全")


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


def _probe_dataset(log: TrainLogger, input_dir: Path, files: list[Path]):
    total_mb = sum(p.stat().st_size for p in files) / 1024 ** 2
    by_ext = {}
    for p in files:
        by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1
    log.log(f"素材目录    : {input_dir.resolve()}")
    log.log(f"音频文件    : {len(files)} 个，共 {total_mb:.1f} MB")
    log.log("格式分布    : " + ", ".join(f"{k}={v}" for k, v in sorted(by_ext.items())))
    if len(files) < 10:
        log.log("素材偏少（<10 个文件），音质与稳定性会明显下降，建议 30 分钟以上干净人声", "WARN")


# ── 4. 问答向导 ──────────────────────────────────────────────────────
def _detect_ckpt_epoch(exp_dir: Path) -> int:
    from rvc.train.ckpt_utils import checkpoints_dir as _ckpt_dir
    from rvc.train.ckpt_utils import checkpoint_epoch, latest_checkpoint_path

    directory = _ckpt_dir(exp_dir)
    g = latest_checkpoint_path(str(directory), "G") or latest_checkpoint_path(str(exp_dir), "G")
    d = latest_checkpoint_path(str(directory), "D") or latest_checkpoint_path(str(exp_dir), "D")
    if not g or not d:
        return 0
    return min(checkpoint_epoch(g), checkpoint_epoch(d))


def _clear_checkpoints(log: TrainLogger, exp_dir: Path):
    from rvc.train.ckpt_utils import checkpoints_dir as _ckpt_dir

    removed = 0
    for directory in (_ckpt_dir(exp_dir), exp_dir):
        if not directory.is_dir():
            continue
        for pattern in ("G_*.pth", "D_*.pth"):
            for path in directory.glob(pattern):
                path.unlink()
                removed += 1
    if removed:
        log.log(f"已删除旧 checkpoint: {removed} 个（从头开始训练）", "WARN")


def run_wizard(log: TrainLogger, device: str, fp16: bool, vram_gb: float) -> dict:
    cfg = {"device": device, "fp16": fp16}
    argv_dir = sys.argv[1] if len(sys.argv) > 1 else ""

    log.section("参数配置（直接回车 = 使用默认值）")

    # 1) 数据集路径
    def _dir_ok(text):
        path = Path(text).expanduser()
        files = _scan_audio(path)
        if not path.is_dir():
            return False, f"目录不存在: {path}"
        if not files:
            return False, f"目录里没有音频文件（支持 {', '.join(sorted(AUDIO_EXTS))}）: {path}"
        return True, ""

    input_dir = Path(ask("数据集路径", argv_dir, check=_dir_ok)).expanduser()
    files = _scan_audio(input_dir)

    # 2) 实验名
    exp_name = ask("实验名（权重与日志都放在 logs/<实验名>/ 下）", input_dir.name)
    exp_dir = Path("logs") / exp_name
    log.attach(exp_dir / "autodl_train.log")
    log.log(f"日志文件    : {log.log_file.resolve()}")
    _probe_dataset(log, input_dir, files)

    # 3) 已有 checkpoint → 续训？
    ckpt_epoch = _detect_ckpt_epoch(exp_dir)
    fresh_start = False
    if ckpt_epoch > 0:
        log.log(f"发现已有 checkpoint：已训练到 epoch {ckpt_epoch}")
        resume = ask_yes("继续训练（选 n 则删除旧 checkpoint 从头开始）？", default=True)
        fresh_start = not resume
    elif exp_dir.exists():
        log.log(f"实验目录已存在（logs/{exp_name}）但无 checkpoint，将复用其中已有的切片/特征")

    # 4) 采样率（权重体检依赖它）
    sr = ask("采样率 40k / 48k", "48k", cast=_as_sr, check=lambda v: (v in ("40k", "48k"), "只能填 40k 或 48k"))
    sr_hz = 48000 if sr == "48k" else 40000

    log.section("环境体检 2/2：ffmpeg / 权重")
    _probe_ffmpeg(log, files)
    _probe_assets(log, sr_hz)

    # 5) 训练轮次
    def _epochs_ok(value):
        if value < 1:
            return False, "至少 1 轮"
        if ckpt_epoch and not fresh_start and value <= ckpt_epoch:
            return False, f"已有 checkpoint 在 epoch {ckpt_epoch}，总轮次必须大于它（那是目标终点，不是新增轮数）"
        return True, ""

    epochs = ask("训练轮次（总轮次；中途 Ctrl+C 会保存后再退出）", "100", cast=_as_int, check=_epochs_ok)

    # 6) batch size
    batch = ask(
        "batch size",
        str(_suggest_batch(vram_gb, sr_hz)),
        cast=_as_int,
        check=lambda v: (v >= 1, "至少 1"),
    )

    log.plain()
    log.plain("以下为进阶参数，不清楚就一路回车：")
    save_every = ask("每多少轮保存一次", "20", cast=_as_int, check=lambda v: (v >= 1, "至少 1"))
    early_stop = ask("连续多少轮 loss 无改善就自动停止（0=关闭）", "30", cast=_as_int, check=lambda v: (v >= 0, "不能为负"))
    per = ask("切片时长（秒）", "3.7", cast=_as_float, check=lambda v: (0.5 <= v <= 30, "建议 1~15 秒"))
    keep_models = ask("assets/models 里保留最近几个导出模型", "2", cast=_as_int, check=lambda v: (v >= 1, "至少 1"))
    keep_ckpts = ask("保留最近几组 checkpoint（每组含 G+D+优化器状态，约 0.8GB）", "1", cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    sr_k = sr_hz // 1000
    pretrain_g = PROJECT_ROOT / "assets" / "pretrained" / f"f0G{sr_k}k.pth"
    pretrain_d = PROJECT_ROOT / "assets" / "pretrained" / f"f0D{sr_k}k.pth"

    cfg.update(
        input_dir=str(input_dir),
        exp_name=exp_name,
        exp_dir=exp_dir,
        sr=sr,
        sr_hz=sr_hz,
        epochs=epochs,
        batch_size=batch,
        save_every=save_every,
        early_stop=early_stop,
        per=per,
        keep_models=keep_models,
        keep_ckpts=keep_ckpts,
        fresh_start=fresh_start,
        ckpt_epoch=ckpt_epoch,
        pretrain_g=str(pretrain_g) if pretrain_g.exists() else "",
        pretrain_d=str(pretrain_d) if pretrain_d.exists() else "",
    )
    return cfg


def print_summary(log: TrainLogger, cfg: dict):
    log.section("配置确认")
    rows = [
        ("数据集", cfg["input_dir"]),
        ("实验名", f"logs/{cfg['exp_name']}" + (f"（从 epoch {cfg['ckpt_epoch'] + 1} 续训）" if cfg["ckpt_epoch"] and not cfg["fresh_start"] else "")),
        ("采样率", cfg["sr"]),
        ("总轮次", cfg["epochs"]),
        ("batch size", cfg["batch_size"]),
        ("保存间隔", f"每 {cfg['save_every']} 轮"),
        ("早停", f"{cfg['early_stop']} 轮无改善" if cfg["early_stop"] else "关闭"),
        ("切片时长", f"{cfg['per']}s"),
        ("保留导出模型", f"最近 {cfg['keep_models']} 个"),
        ("保留 checkpoint", f"最近 {cfg['keep_ckpts']} 组"),
        ("设备", f"{cfg['device']}  fp16={cfg['fp16']}"),
        ("底模", cfg["pretrain_g"] or "无（从零训练）"),
    ]
    for key, value in rows:
        log.plain(f"  {key:<16} {value}")
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
    state = {"last": -1}

    def cb(done, total):
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct != state["last"] and (pct % 5 == 0 or done == total):
            state["last"] = pct
            log.log(f"{label}: {done}/{total} ({pct}%)", "进度")

    return cb


def step_preprocess(log: TrainLogger, cfg: dict):
    from rvc.train.preprocess import PreProcessor

    log.section("步骤 1/4 预处理音频（切片 + 重采样）")
    log.log(f"切片时长 {cfg['per']}s，目标采样率 {cfg['sr_hz']} Hz")
    t0 = time.time()
    processor = PreProcessor(cfg["input_dir"], str(cfg["exp_dir"]), cfg["sr_hz"], per=cfg["per"])
    count = processor.run(_make_progress(log, "预处理"))
    log.log(f"预处理完成：{count} 个源文件，用时 {time.time() - t0:.1f}s")
    gt = sorted((cfg["exp_dir"] / "0_gt_wavs").glob("*.wav"))
    log.log(f"生成切片：{len(gt)} 条")
    if len(gt) < 50:
        log.log("切片数偏少（<50），训练容易过拟合，建议补充素材", "WARN")


def step_f0(log: TrainLogger, cfg: dict):
    from rvc.train.extract_f0 import TrainF0Extractor

    log.section("步骤 2/4 提取 F0（RMVPE）")
    t0 = time.time()
    extractor = TrainF0Extractor(cfg["device"], cfg["fp16"])
    n = extractor.run(str(cfg["exp_dir"]), _make_progress(log, "F0"), stop_check=lambda: STOP.requested)
    log.log(f"F0 提取完成：{n} 条，用时 {time.time() - t0:.1f}s")


def step_feature(log: TrainLogger, cfg: dict):
    from rvc.train.extract_feature import HuBERTExtractor

    log.section("步骤 3/4 提取 HuBERT 特征")
    t0 = time.time()
    extractor = HuBERTExtractor(cfg["device"], cfg["fp16"])
    n = extractor.run(str(cfg["exp_dir"]), _make_progress(log, "特征"), stop_check=lambda: STOP.requested)
    log.log(f"特征提取完成：{n} 条，用时 {time.time() - t0:.1f}s")


def _features_ready(exp_dir: Path) -> bool:
    gt = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
    if gt == 0:
        return False
    f0 = len(list((exp_dir / "2a_f0").glob("*.npy")))
    f0nsf = len(list((exp_dir / "2b-f0nsf").glob("*.npy")))
    feat = len(list((exp_dir / "3_feature768").glob("*.npy")))
    return f0 >= gt and f0nsf >= gt and feat >= gt


def _gpu_mem_used() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        return 0.0


def step_train(log: TrainLogger, cfg: dict):
    from rvc.train.preprocess import generate_filelist
    from rvc.train.trainer import TrainConfig, Trainer

    log.section("步骤 4/4 训练")
    _, count = generate_filelist(str(cfg["exp_dir"]))
    log.log(f"训练样本数: {count}")
    if count == 0:
        raise RuntimeError("没有可训练样本（特征或 F0 缺失），请删除 logs/<实验名> 重跑")

    for name in ("pretrain_g", "pretrain_d"):
        path = cfg[name]
        if path and not Path(path).exists():
            raise RuntimeError(f"预训练模型不存在: {path}")

    train_config = TrainConfig(
        exp_dir=str(cfg["exp_dir"]),
        sr=cfg["sr_hz"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        save_every_epoch=cfg["save_every"],
        learning_rate=1e-4,
        pretrain_g=cfg["pretrain_g"],
        pretrain_d=cfg["pretrain_d"],
        fp16_run=cfg["fp16"],
        device=cfg["device"],
        early_stop_patience=cfg["early_stop"],
        keep_ckpts=cfg["keep_ckpts"],
        keep_models=cfg["keep_models"],
    )

    state = {"last": None, "t_epoch": time.time(), "batches": 0}

    def on_batch(epoch, batch, total):
        state["batches"] += 1

    def on_loss(info):
        state["last"] = info
        if info["batch"] == 1:
            # 首个 epoch 的计时从 setup（加载底模）就开始了，重置一次，速度才准
            state["t_epoch"] = time.time()
        if info["batch"] % 20 == 0 or info["batch"] == 1:
            log.log(
                f"e{info['epoch']:>4} b{info['batch']:>5} | "
                f"D {info['loss_d']:.4f} G {info['loss_g']:.4f} "
                f"Mel {info['loss_mel']:.4f} KL {info['loss_kl']:.4f} FM {info['loss_fm']:.4f}",
                "STEP",
            )

    def on_epoch(epoch, total):
        # loss 由 Trainer 自己按 epoch 平均值落盘 train.log，这里只报进度/速度/显存
        secs = time.time() - state["t_epoch"]
        state["t_epoch"] = time.time()
        left = secs * (total - epoch)
        eta = time.strftime("%Hh%Mm", time.gmtime(left)) if left > 0 else "--"
        mem = f" | 显存 {_gpu_mem_used():.1f}GB" if cfg["device"].startswith("cuda") else ""
        log.log(f"epoch {epoch:>4}/{total} | {secs:.1f}s/epoch 剩余 {eta}{mem}", "EPOCH")

    trainer = Trainer(train_config, on_epoch, log.log, on_loss, on_batch)
    STOP.trainer = trainer
    t0 = time.time()
    try:
        output = trainer.train()
    finally:
        trainer.cleanup()
        STOP.trainer = None

    log.log(f"训练结束，总用时 {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - t0))}")
    if cfg["device"].startswith("cuda"):
        log.log(f"显存峰值  : {_gpu_mem_used():.2f} GB")
    log.log(f"模型已导出: {output}")
    return output


# ── 6. 主流程 ────────────────────────────────────────────────────────
def main() -> int:
    # Trainer 里 scheduler.step() 先于 optimizer.step() 的既有写法会每轮刷两条警告，
    # 属于原代码行为，这里只在脚本侧静音，不改源文件
    warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")

    log = TrainLogger()
    try:
        log.section("RVC 云 GPU 训练向导")
        log.log(f"项目根目录  : {PROJECT_ROOT}")
        log.log(f"Python      : {sys.version.split()[0]}")

        log.section("环境体检 1/2：依赖 / GPU")
        if _probe_packages(log):
            return EXIT_ENV
        device, fp16, vram_gb = _probe_gpu(log)
        log.log(f"最终使用   : device={device}  fp16={fp16}")

        cfg = run_wizard(log, device, fp16, vram_gb)
        print_summary(log, cfg)

        log.plain("按 Enter 开始训练（Ctrl+C 取消；训练中 Ctrl+C = 本轮跑完保存后退出）")
        try:
            input("  ▶ ")
        except (EOFError, KeyboardInterrupt):
            log.log("已取消", "WARN")
            return EXIT_OK

        _install_signal_handlers(log)
        exp_dir = cfg["exp_dir"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        if cfg["fresh_start"]:
            _clear_checkpoints(log, exp_dir)

        from rvc.train.preprocess import manifest_diff_reason

        reason = manifest_diff_reason(exp_dir, cfg["input_dir"], cfg["sr_hz"], cfg["per"])
        if reason:
            log.log(f"需要重新预处理：{reason}")
            step_preprocess(log, cfg)
        else:
            log.log("切片与素材一致，跳过预处理（要重建请删 logs/%s/manifest.json）" % cfg["exp_name"])

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
        log.log(f"总耗时 {log.elapsed()}")
        log.log(f"导出模型: {(PROJECT_ROOT / 'assets' / 'models').resolve()}")
        log.log(f"checkpoint: {(exp_dir / '4_checkpoints').resolve()}")
        log.log(f"训练日志: {(exp_dir / 'train.log').resolve()}")
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
