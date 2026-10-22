"""训练参数向导：交互式收集所有训练参数并输出配置字典。

提问面固定 11 项：数据集路径、产物根目录、实验名、采样率、特征器、
batch size、学习率、训练轮次、保存间隔、保留 checkpoint 组数、是否用底模。
切片时长/keep_models 是固定默认值，不提问——不要再把内部常量变成新提问项。

提问一律「平铺」，每项都在问题旁用括号标出说明、可选项与默认值，直接回车即取默认值
（默认值 = 上次的答案，见 `rvc.train.cli.defaults`）。
"""
from pathlib import Path

from rvc.runtime.paths import PRETRAINED_ROOT, TRAIN_CLI_DEFAULTS_FILE

from rvc.train.cli import (
    AUDIO_EXTS,
    BAR,
    DEFAULTS,
    output_dirs,
)
from rvc.train.cli.defaults import load_defaults, save_defaults
from rvc.train.cli.envcheck import (
    probe_dataset,
    _probe_disk,
    _require_ffmpeg,
    _scan_audio,
    _suggest_batch,
)
from rvc.models.hubert import HUBERT_VARIANTS
from rvc.runtime.paths import parse_sr
from rvc.train.cli.logger import TrainLogger
from rvc.train.cli.prompts import _as_float, _as_int, _as_sr, ask, ask_yes

# 采样率可选项（同时作为 check 的白名单）；特征器清单单源在 rvc.models.hubert
SR_OPTIONS = ("40k", "48k")
HUBERT_OPTIONS = HUBERT_VARIANTS


def _detect_ckpt_epoch(exp_dir: Path) -> int:
    from rvc.train.checkpoint import checkpoints_dir as _ckpt_dir
    from rvc.train.checkpoint import checkpoint_epoch, latest_checkpoint_path

    directory = _ckpt_dir(exp_dir)
    g = latest_checkpoint_path(str(directory), "G") or latest_checkpoint_path(str(exp_dir), "G")
    d = latest_checkpoint_path(str(directory), "D") or latest_checkpoint_path(str(exp_dir), "D")
    if not g or not d:
        return 0
    return min(checkpoint_epoch(g), checkpoint_epoch(d))


def run_wizard(log: TrainLogger, device: str, fp16: bool, vram_gb: float, argv_dir: str,
               ffmpeg_exe: str | None = None) -> dict:
    """交互收集训练参数。

    argv_dir: 命令行给出的数据集路径（第一个非选项参数），作为数据集提问的默认值。
    ffmpeg_exe: 环境体检阶段定位的 ffmpeg 路径，随 cfg 下发给后续所有解码调用；
    None 表示未定位到（仅 wav/flac/ogg 素材可继续）。

    各提问项的默认值来自上次运行的存档（`rvc.train.cli.defaults`），
    问完后本次答案会被存回去，作为下次的默认值。
    """
    cfg = {"device": device, "fp16": fp16, "ffmpeg": ffmpeg_exe}
    defaults = load_defaults()

    log.section("第 3 步 / 训练参数（直接回车 = 使用默认值，默认值 = 上次的答案）")

    # 1) 数据集路径（命令行参数 > 上次的路径 > 必填）
    def _dir_ok(text):
        path = Path(text).expanduser()
        if not path.is_dir():
            return False, f"目录不存在: {path}"
        if not _scan_audio(path):
            return False, f"目录里没有音频文件（支持 {', '.join(sorted(AUDIO_EXTS))}）: {path}"
        return True, ""

    input_dir = Path(ask("数据集路径", argv_dir or defaults["input_dir"],
                         check=_dir_ok)).expanduser()
    files = _scan_audio(input_dir)
    probe_dataset(log, input_dir, files, ffmpeg_path=ffmpeg_exe)
    _require_ffmpeg(log, files, ffmpeg_exe=ffmpeg_exe)

    # 2) 产物根目录（实验目录与导出模型目录都在其下；默认按平台自适应）
    root_default = Path(defaults["output_root"])

    def _root_ok(text):
        path = Path(text).expanduser()
        if path.exists() and not path.is_dir():
            return False, f"已存在同名文件: {path}"
        return True, ""

    output_root = Path(ask("产物根目录", str(root_default), check=_root_ok,
                           note="切片/特征/checkpoint/导出模型都在其下")).expanduser()
    _probe_disk(log, output_root)

    # 3) 实验名（决定产物落点，先拿到才好把日志搬过去）
    exp_name = ask("实验名", defaults["exp_name"])
    exp_dir, model_dir = output_dirs(output_root, exp_name)
    # 必须在 reattach 之前记录：reattach 会创建实验目录，之后 exists() 恒为真
    exp_existed = exp_dir.exists()
    log.reattach(exp_dir / "train_cli.log")
    log.log(f"产物根目录  : {output_root}")
    log.log(f"实验目录    : {exp_dir}")
    log.log(f"模型目录    : {model_dir}")
    log.log(f"日志文件    : {log.log_file}")

    # 续训检测（自动进行，不属 11 项提问：检测到就续训；要重来请手动删掉 checkpoint 目录）
    ckpt_epoch = _detect_ckpt_epoch(exp_dir)
    if ckpt_epoch:
        log.log(f"检测到已有 checkpoint（已训练到 epoch {ckpt_epoch}）→ 自动续训")
        log.log(f"  要重新训练请先删掉: {exp_dir / '4_checkpoints'}")
    elif exp_existed:
        log.log("实验目录已存在但无 checkpoint，将复用其中已有的切片/特征")

    # 4) 采样率
    def _sr_ok(value):
        if value not in SR_OPTIONS:
            return False, "只能填 40k 或 48k"
        return True, ""

    sr = ask("采样率", defaults["sr"], cast=_as_sr, check=_sr_ok, options=SR_OPTIONS,
             note="40k 更省显存、更快；48k 高频细节更多")
    sr_hz = parse_sr(sr)
    if sr == "40k":
        log.log("40k：显存占用更小、训练更快，高频细节略少于 48k", "WARN")

    # 5) HuBERT 特征器（训练/推理必须一致，权重在 assets/hubert/{base,chinese}；
    #    不一致 → 音色发闷、咬字怪异）
    def _hubert_ok(value):
        if value not in HUBERT_OPTIONS:
            return False, "只能填 base 或 chinese"
        return True, ""

    hubert = ask("HuBERT 特征器", defaults["hubert"], check=_hubert_ok, options=HUBERT_OPTIONS,
                 note="chinese = 腾讯 chinese-hubert-base，中文音素/声调更准")
    if hubert == "chinese":
        log.log("chinese 特征器：推理时模型卡牌上的「特征器」必须同样选 chinese", "WARN")

    # 6) batch size（上次填过就用上次的，没填过按显存现算）
    suggested = _suggest_batch(vram_gb, sr_hz)
    batch_default = defaults["batch_size"] or suggested
    batch = ask("batch size", str(batch_default), cast=_as_int,
                check=lambda v: (v >= 1, "至少 1"),
                note=f"按 {vram_gb:.1f}GB 显存建议 {suggested}")

    # 7) 学习率
    lr = ask(
        "学习率",
        f"{defaults['lr']:g}",
        cast=_as_float,
        check=lambda v: (0 < v <= 0.1, "需要在 0 ~ 0.1 之间"),
        note="1e-4=0.0001；太大易炸、太小收敛慢",
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

    # 续训时「总轮次」必须大于已训到的 epoch：存档值若已不满足（上次正好训到该轮），
    # 默认值取 2 倍已训轮次，否则非交互输入（管道喂 stdin）会卡在反复重问上
    epochs_default = ckpt_epoch * 2 if ckpt_epoch >= defaults["epochs"] else defaults["epochs"]
    epochs = ask("训练轮次", str(epochs_default), cast=_as_int, check=_epochs_ok,
                 note="填目标终点；已训到的轮次会自动续训")

    # 9) 保存间隔
    save_every = ask("每多少轮保存一次", str(defaults["save_every"]), cast=_as_int,
                     check=lambda v: (v >= 1, "至少 1"))

    # 10) 保留 checkpoint 组数
    keep_ckpts = ask("保留最近几组 checkpoint", str(defaults["keep_ckpts"]), cast=_as_int,
                     check=lambda v: (v >= 1, "至少 1"),
                     note="每组含优化器状态，是体积最大的产物")

    # 11) 是否使用预训练底模
    use_pretrained = ask_yes("是否使用预训练模型", defaults["use_pretrained"],
                             note="No = 随机初始化")
    if not use_pretrained:
        log.log("已选择不用预训练底模 → 从随机权重开始训练，需要更多轮次，效果通常不如用底模", "WARN")

    sr_k = sr_hz // 1000
    pretrain_g = PRETRAINED_ROOT / f"f0G{sr_k}k.pth"
    pretrain_d = PRETRAINED_ROOT / f"f0D{sr_k}k.pth"

    cfg.update(
        input_dir=str(input_dir),
        output_root=str(output_root),
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

    # 本次答案存为下次默认值（即使随后在「开始训练？」处取消也已保存）
    save_defaults(cfg)
    log.log(f"本次参数已存为下次默认值: {TRAIN_CLI_DEFAULTS_FILE}")
    log.log("  （想要恢复内置默认值，删掉该文件即可）")

    return cfg


def print_summary(log: TrainLogger, cfg: dict):
    log.section("配置确认")
    ckpt = cfg["ckpt_epoch"]
    rows = [
        ("数据集", cfg["input_dir"]),
        ("产物根目录", cfg["output_root"]),
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
        ("底模", cfg["pretrain_g"] or ("无（随机初始化）" if not cfg.get("use_pretrained", True) else "无（未找到底模）")),
    ]
    for key, value in rows:
        log.plain(f"  {key:<14} {value}")
        log.log(f"配置 {key}: {value}")
    log.plain(BAR)
