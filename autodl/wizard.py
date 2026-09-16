"""训练参数向导：交互式收集所有训练参数并输出配置字典。"""
from pathlib import Path

from rvc.runtime.paths import PRETRAINED_ROOT

from autodl import AUDIO_EXTS, BAR, DEFAULTS, LOGS_ROOT, MODELS_ROOT
from autodl.envcheck import _probe_dataset, _require_ffmpeg, _scan_audio, _suggest_batch
from autodl.logger import TrainLogger
from autodl.prompts import _as_float, _as_int, _as_sr, ask, ask_yes


def _detect_ckpt_epoch(exp_dir: Path) -> int:
    from rvc.train.checkpoint import checkpoints_dir as _ckpt_dir
    from rvc.train.checkpoint import checkpoint_epoch, latest_checkpoint_path

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

    # 5) HuBERT 特征器
    def _hubert_ok(value):
        if value not in ("base", "chinese"):
            return False, "只能填 base 或 chinese"
        return True, ""

    hubert = ask("HuBERT 特征器 base / chinese", DEFAULTS["hubert"], check=_hubert_ok)
    if hubert == "chinese":
        log.log("chinese = 腾讯 chinese-hubert-base（WenetSpeech 中文预训练，中文音素/声调更准）", "WARN")
        log.log("  推理时模型卡牌上的「特征器」必须同样选 chinese", "WARN")

    # 6) batch size
    batch = ask("batch size", str(_suggest_batch(vram_gb, sr_hz)), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    # 7) 学习率
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

    # 8) 总轮次
    def _epochs_ok(value):
        if value < 1:
            return False, "至少 1 轮"
        if ckpt_epoch and value <= ckpt_epoch:
            return False, f"已训到 epoch {ckpt_epoch}，总轮次必须大于它（填的是目标终点，不是新增轮数）"
        return True, ""

    epochs = ask("训练轮次", str(DEFAULTS["epochs"]), cast=_as_int, check=_epochs_ok)

    # 9) 保存间隔
    save_every = ask("每多少轮保存一次", str(DEFAULTS["save_every"]), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    # 10) 保留 checkpoint 组数
    keep_ckpts = ask("保留最近几组 checkpoint", str(DEFAULTS["keep_ckpts"]), cast=_as_int, check=lambda v: (v >= 1, "至少 1"))

    # 11) 是否使用预训练底模
    use_pretrained = ask_yes("是否使用预训练模型（No = 随机初始化）", DEFAULTS["use_pretrained"])
    if not use_pretrained:
        log.log("已选择不用预训练底模 → 从随机权重开始训练，需要更多轮次，效果通常不如用底模", "WARN")

    sr_k = sr_hz // 1000
    pretrain_g = PRETRAINED_ROOT / f"f0G{sr_k}k.pth"
    pretrain_d = PRETRAINED_ROOT / f"f0D{sr_k}k.pth"

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
        hubert=hubert,
        use_pretrained=use_pretrained,
        pretrain_g=str(pretrain_g) if (use_pretrained and pretrain_g.exists()) else "",
        pretrain_d=str(pretrain_d) if (use_pretrained and pretrain_d.exists()) else "",
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
        ("特征器", cfg["hubert"]),
        ("总轮次", str(cfg["epochs"]) + (f"（从 epoch {ckpt + 1} 续训）" if ckpt else "")),
        ("batch size", cfg["batch_size"]),
        ("学习率", f"{cfg['lr']:g}"),
        ("保存间隔", f"每 {cfg['save_every']} 轮"),
        ("切片时长", f"{cfg['per']}s（固定）"),
        ("保留 checkpoint", f"最近 {cfg['keep_ckpts']} 组"),
        ("设备", f"{cfg['device']}  fp16={cfg['fp16']}"),
        ("底模", cfg["pretrain_g"] or ("无（随机初始化）" if not cfg.get("use_pretrained", True) else "无（未上传底模）")),
    ]
    for key, value in rows:
        log.plain(f"  {key:<14} {value}")
        log.log(f"配置 {key}: {value}")
    log.plain(BAR)
