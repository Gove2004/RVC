"""训练参数向导：交互式收集所有训练参数并输出配置字典。

提问一律「平铺」——没有进阶闸门，全部参数依次问一遍；每项都在问题旁用括号标出
可选项与默认值，直接回车即取默认值（默认值 = 上次的答案，见 `rvc.train.cli.defaults`）。
"""
from pathlib import Path

from rvc.runtime.paths import PRETRAINED_ROOT, TRAIN_CLI_DEFAULTS_FILE

from rvc.train.cli import (
    AUDIO_EXTS,
    BAR,
    output_dirs,
)
from rvc.train.cli.defaults import load_defaults, save_defaults
from rvc.train.cli.envcheck import (
    _probe_dataset,
    _probe_disk,
    _require_ffmpeg,
    _scan_audio,
    _suggest_batch,
)
from rvc.train.cli.logger import TrainLogger
from rvc.train.cli.prompts import _as_float, _as_int, _as_pair, _as_sr, ask, ask_yes

# 采样率 / 特征器 / 是否用底模 的可选项（同时作为 check 的白名单，避免两处各写一份）
SR_OPTIONS = ("40k", "48k")
HUBERT_OPTIONS = ("base", "chinese")


def _num(value) -> str:
    """数值 → 提问默认值文本（48.0 → 48；0.999875 原样；整数原样）。"""
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _pick(saved, json_value):
    """默认值：存档优先（None = 存档里没有这一项），否则用当前采样率 json 的值。"""
    return json_value if saved is None else saved


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
               ffmpeg_exe: str = "") -> dict:
    """交互收集训练参数。

    argv_dir: 命令行给出的数据集路径（第一个非选项参数），作为数据集提问的默认值。
    ffmpeg_exe: 环境体检阶段定位的 ffmpeg 路径，随 cfg 下发给后续所有解码调用；
    空串表示未定位到（仅 wav/flac/ogg 素材可继续）。

    各提问项的默认值来自上次运行的存档（`rvc.train.cli.defaults`），
    问完后本次答案会被存回去，作为下次的默认值。
    """
    cfg = {"device": device, "fp16": fp16, "ffmpeg": ffmpeg_exe}
    defaults = load_defaults()

    log.section("第 3 步 / 训练参数（直接回车 = 使用默认值，默认值 = 上次的答案）")

    # ── 1) 数据集路径（命令行参数 > 上次的路径 > 必填）──
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
    _probe_dataset(log, input_dir, files, ffmpeg_path=ffmpeg_exe or None)
    _require_ffmpeg(log, files)

    # ── 2) 产物根目录（实验目录与导出模型目录都在其下；默认按平台自适应）──
    root_default = Path(defaults["output_root"])

    def _root_ok(text):
        path = Path(text).expanduser()
        if path.exists() and not path.is_dir():
            return False, f"已存在同名文件: {path}"
        return True, ""

    output_root = Path(ask("产物根目录", str(root_default), check=_root_ok,
                           note="切片/特征/checkpoint/导出模型都在其下")).expanduser()
    _probe_disk(log, output_root)

    # ── 3) 实验名（决定产物落点，先拿到才好把日志搬过去）──
    exp_name = ask("实验名", defaults["exp_name"])
    exp_dir, model_dir = output_dirs(output_root, exp_name)
    # 必须在 reattach 之前记录：reattach 会创建实验目录，之后 exists() 恒为真
    exp_existed = exp_dir.exists()
    log.reattach(exp_dir / "train_cli.log")
    log.log(f"产物根目录  : {output_root}")
    log.log(f"实验目录    : {exp_dir}")
    log.log(f"模型目录    : {model_dir}")
    log.log(f"日志文件    : {log.log_file}")

    # 4) 续训检测（不提问：检测到就续训；要重来请手动删掉 checkpoint 目录）
    ckpt_epoch = _detect_ckpt_epoch(exp_dir)
    if ckpt_epoch:
        log.log(f"检测到已有 checkpoint（已训练到 epoch {ckpt_epoch}）→ 自动续训")
        log.log(f"  要重新训练请先删掉: {exp_dir / '4_checkpoints'}")
    elif exp_existed:
        log.log("实验目录已存在但无 checkpoint，将复用其中已有的切片/特征")

    # ── 5) 采样率 ──
    def _sr_ok(value):
        if value not in SR_OPTIONS:
            return False, "只能填 40k 或 48k"
        return True, ""

    sr = ask("采样率", defaults["sr"], cast=_as_sr, check=_sr_ok, options=SR_OPTIONS,
             note="40k 更省显存、更快；48k 高频细节更多")
    sr_hz = 48000 if sr == "48k" else 40000
    if sr == "40k":
        log.log("40k：显存占用更小、训练更快，高频细节略少于 48k", "WARN")

    # 训练底参（c_mel/c_kl/lr_decay/seed/segment_size/betas/eps）的默认值唯一来源
    # 是当前采样率对应的 train_config.json —— 采样率定了才能读
    from rvc.train.checkpoint import load_train_json

    train_json = load_train_json(sr_hz)
    jtrain = train_json["train"]
    hop_length = train_json["data"]["hop_length"]

    # ── 6) HuBERT 特征器 ──
    def _hubert_ok(value):
        if value not in HUBERT_OPTIONS:
            return False, "只能填 base 或 chinese"
        return True, ""

    hubert = ask("HuBERT 特征器", defaults["hubert"], check=_hubert_ok, options=HUBERT_OPTIONS,
                 note="chinese = 腾讯 chinese-hubert-base，中文音素/声调更准")
    if hubert == "chinese":
        log.log("chinese 特征器：推理时模型卡牌上的「特征器」必须同样选 chinese", "WARN")

    # ── 7) batch size（上次填过就用上次的，没填过按显存现算）──
    suggested = _suggest_batch(vram_gb, sr_hz)
    batch_default = defaults["batch_size"] or suggested
    batch = ask("batch size", str(batch_default), cast=_as_int,
                check=lambda v: (1 <= v <= 4096, "需要在 1 ~ 4096 之间"),
                note=f"按 {vram_gb:.1f}GB 显存建议 {suggested}")

    # ── 8) 学习率 ──
    lr = ask(
        "学习率",
        _num(defaults["lr"]),
        cast=_as_float,
        check=lambda v: (0 < v <= 0.1, "需要在 0 ~ 0.1 之间"),
        note="1e-4=0.0001；太大易炸、太小收敛慢",
    )
    if lr > 5e-4:
        log.log("学习率偏大，GAN 容易崩（loss 变 nan）；新手建议 1e-4 附近", "WARN")
    elif lr < 1e-5:
        log.log("学习率偏小，100 轮大概率训不出东西", "WARN")

    # ── 9) 总轮次 ──
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

    # ── 10) 保存间隔 ──
    save_every = ask("每多少轮保存一次", str(defaults["save_every"]), cast=_as_int,
                     check=lambda v: (v >= 1, "至少 1"))

    # ── 11) 保留 checkpoint 组数 ──
    keep_ckpts = ask("保留最近几组 checkpoint", str(defaults["keep_ckpts"]), cast=_as_int,
                     check=lambda v: (v >= 1, "至少 1"),
                     note="每组含优化器状态，是体积最大的产物")

    # ── 12) 是否使用预训练底模 ──
    use_pretrained = ask_yes("是否使用预训练模型", defaults["use_pretrained"],
                             note="No = 随机初始化")
    if not use_pretrained:
        log.log("已选择不用预训练底模 → 从随机权重开始训练，需要更多轮次，效果通常不如用底模", "WARN")

    # ══════════════ 预处理 / 切片 ══════════════
    log.log("—— 预处理 / 切片（改任一项都会重新切片，旧切片与 checkpoint 会被清除）——")

    # ── 13) 切片时长 ──
    def _per_ok(value):
        if not 1.0 <= value <= 30.0:
            return False, "需要在 1 ~ 30 秒之间（小于 1 秒切不出任何片段）"
        return True, ""

    per = ask("切片时长", _num(defaults["per"]), cast=_as_float, check=_per_ok,
              note="单位秒；单条训练切片的长度，常用 3 ~ 10")

    # ── 14) 切片重叠 ──
    def _overlap_ok(value):
        if not 0 <= value < per:
            return False, f"需要在 0 ~ {_num(per)} 秒之间（必须小于切片时长）"
        return True, ""

    overlap = ask("相邻切片重叠", _num(defaults["overlap"]), cast=_as_float,
                  check=_overlap_ok, note="单位秒；保证切片边界处的字不被切碎")

    # ── 15) 高通截止 ──
    highpass = ask("预处理高通截止", _num(defaults["highpass"]), cast=_as_float,
                   check=lambda v: (10 <= v <= 500, "需要在 10 ~ 500 Hz 之间"),
                   note="单位 Hz；滤掉素材里的直流与低频隆隆声")

    # ── 16~20) 静音切分 ──
    slice_threshold = ask("静音判定阈值", _num(defaults["slice_threshold"]), cast=_as_float,
                          check=lambda v: (-100 <= v <= 0, "需要在 -100 ~ 0 dB 之间"),
                          note="单位 dB；低于它就当作静音，在静音中间下刀")
    slice_min_length = ask("最短切片", _num(defaults["slice_min_length"]), cast=_as_int,
                           check=lambda v: (10 <= v <= 60000, "需要在 10 ~ 60000 毫秒之间"),
                           note="单位毫秒；短于它的段落不会被单独切出来")
    slice_min_interval = ask("最短静音间隔", _num(defaults["slice_min_interval"]), cast=_as_int,
                             check=lambda v: (0 <= v <= 60000, "需要在 0 ~ 60000 毫秒之间"),
                             note="单位毫秒；短于此长度的静音不当作切点")
    slice_max_sil_kept = ask("单次最多保留静音", _num(defaults["slice_max_sil_kept"]), cast=_as_int,
                             check=lambda v: (0 <= v <= 60000, "需要在 0 ~ 60000 毫秒之间"),
                             note="单位毫秒；切点附近保留多少静音，0 = 全部裁掉")
    slice_hop_size = ask("静音分析帧步长", _num(defaults["slice_hop_size"]), cast=_as_int,
                         check=lambda v: (1 <= v <= 1000, "需要在 1 ~ 1000 毫秒之间"),
                         note="单位毫秒；越小对静音边界越敏感，帧长 = 2 倍步长")

    # ══════════════ 训练底参（默认值来自 train_config.json）══════════════
    log.log(f"—— 训练底参（默认值来自 {sr} 的 train_config.json，改错会影响收敛）——")

    c_mel = ask("梅尔损失权重 c_mel", _num(_pick(defaults["c_mel"], jtrain["c_mel"])),
                cast=_as_float, check=lambda v: (0 <= v <= 10000, "需要在 0 ~ 10000 之间"),
                note="人声相似度的主项，调大更贴音色但可能糊")
    c_kl = ask("KL 损失权重 c_kl", _num(_pick(defaults["c_kl"], jtrain["c_kl"])),
               cast=_as_float, check=lambda v: (0 <= v <= 1000, "需要在 0 ~ 1000 之间"),
               note="约束音高/时长分布，调大更稳但可能不自然")
    lr_decay = ask("学习率衰减 lr_decay", _num(_pick(defaults["lr_decay"], jtrain["lr_decay"])),
                   cast=_as_float, check=lambda v: (0.5 <= v <= 1.0, "需要在 0.5 ~ 1.0 之间"),
                   note="每轮乘以它；1.0 = 不衰减")
    seed = ask("随机种子 seed", _num(_pick(defaults["seed"], jtrain["seed"])), cast=_as_int,
               check=lambda v: (0 <= v <= 2 ** 31 - 1, "需要在 0 ~ 2147483647 之间"),
               note="同种子 + 同设备 = 可复现")

    def _seg_ok(value):
        if value < 1:
            return False, "至少 1"
        if value % hop_length != 0:
            return False, f"必须是 hop_length（{hop_length}）的整数倍"
        return True, ""

    segment_size = ask("训练片段长度 segment_size", _num(_pick(defaults["segment_size"], jtrain["segment_size"])),
                       cast=_as_int, check=_seg_ok,
                       note=f"单位采样点；{sr} 的 hop_length={hop_length}，必须是它的整数倍")

    # ══════════════ 优化器 ══════════════
    log.log("—— 优化器（AdamW）——")

    betas_raw = _pick(defaults["betas"], jtrain["betas"])
    betas_default = ",".join(_num(v) for v in betas_raw)
    betas = ask("AdamW betas", betas_default, cast=_as_pair,
                check=lambda v: (all(0 <= x < 1 for x in v), "两个数都要在 0 ~ 1 之间（不含 1）"),
                note="两个数、逗号分隔；调大第 1 个会更快跟随当前 batch")
    eps = ask("AdamW eps", _num(_pick(defaults["eps"], jtrain["eps"])), cast=_as_float,
              check=lambda v: (0 < v <= 1, "需要在 0 ~ 1 之间"),
              note="分母防零，一般不用动")

    # ══════════════ 运行时 ══════════════
    log.log("—— 运行时 ——")

    num_workers = ask("取数进程数 num_workers", str(defaults["num_workers"]), cast=_as_int,
                      check=lambda v: (0 <= v <= 256, "需要在 0 ~ 256 之间"),
                      note="0 = 主进程单线程取数，排查卡死时可用")
    rmvpe_threshold = ask("训练侧 RMVPE 阈值", _num(defaults["rmvpe_threshold"]), cast=_as_float,
                          check=lambda v: (0 <= v <= 1, "需要在 0 ~ 1 之间"),
                          note="低于它判为静音；调大→更多静音被判清音")
    keep_models = ask("保留导出模型数", str(defaults["keep_models"]), cast=_as_int,
                      check=lambda v: (v >= 0, "不能为负"),
                      note="0 = 全部保留；>=1 时只留最新 N 个")

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
        hop_length=hop_length,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        save_every=save_every,
        keep_ckpts=keep_ckpts,
        keep_models=keep_models,
        ckpt_epoch=ckpt_epoch,
        hubert=hubert,
        use_pretrained=use_pretrained,
        pretrain_g=str(pretrain_g) if (use_pretrained and pretrain_g.exists()) else "",
        pretrain_d=str(pretrain_d) if (use_pretrained and pretrain_d.exists()) else "",
        # 预处理 / 切片
        per=per,
        overlap=overlap,
        highpass=highpass,
        slice_threshold=slice_threshold,
        slice_min_length=slice_min_length,
        slice_min_interval=slice_min_interval,
        slice_hop_size=slice_hop_size,
        slice_max_sil_kept=slice_max_sil_kept,
        # 训练底参与优化器（经 TrainConfig.overrides 覆盖 json）
        c_mel=c_mel,
        c_kl=c_kl,
        lr_decay=lr_decay,
        seed=seed,
        segment_size=segment_size,
        betas=betas,
        eps=eps,
        # 运行时
        num_workers=num_workers,
        rmvpe_threshold=rmvpe_threshold,
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
        ("保留 checkpoint", f"最近 {cfg['keep_ckpts']} 组"),
        ("设备", f"{cfg['device']}  fp16={cfg['fp16']}"),
        ("底模", cfg["pretrain_g"] or ("无（随机初始化）" if not cfg.get("use_pretrained", True) else "无（未找到底模）")),
        # ── 预处理 / 切片 ──
        ("切片时长", f"{_num(cfg['per'])}s"),
        ("切片重叠", f"{_num(cfg['overlap'])}s"),
        ("高通截止", f"{_num(cfg['highpass'])}Hz"),
        ("静音阈值", f"{_num(cfg['slice_threshold'])}dB"),
        ("最短切片", f"{cfg['slice_min_length']}ms"),
        ("最短静音间隔", f"{cfg['slice_min_interval']}ms"),
        ("保留静音上限", f"{cfg['slice_max_sil_kept']}ms"),
        ("静音帧步长", f"{cfg['slice_hop_size']}ms"),
        # ── 训练底参 / 优化器 ──
        ("c_mel / c_kl", f"{cfg['c_mel']:g} / {cfg['c_kl']:g}"),
        ("lr_decay", f"{cfg['lr_decay']:g}"),
        ("seed", cfg["seed"]),
        ("segment_size", f"{cfg['segment_size']}（hop_length={cfg['hop_length']}）"),
        ("AdamW betas / eps", f"{_num(cfg['betas'][0])},{_num(cfg['betas'][1])} / {cfg['eps']:g}"),
        # ── 运行时 ──
        ("num_workers", cfg["num_workers"]),
        ("RMVPE 阈值(训练)", f"{cfg['rmvpe_threshold']:g}"),
        ("保留导出模型", "全部" if cfg["keep_models"] == 0 else f"最近 {cfg['keep_models']} 个"),
    ]
    for key, value in rows:
        log.plain(f"  {key:<18} {value}")
        log.log(f"配置 {key}: {value}")
    log.plain(BAR)
