"""训练步骤执行：预处理 / F0 提取 / 特征提取 / 训练。"""
import signal
import sys
import time
from pathlib import Path

from autodl import EXIT_RUNTIME
from autodl.logger import TrainLogger
from autodl.prompts import _human_dur, _human_size
from autodl.wizard import _detect_ckpt_epoch


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


def _run_extraction_step(log: TrainLogger, cfg: dict, title: str, label: str,
                         make_extractor, warn: str | None = None) -> int:
    """通用提取步骤：创建 extractor → run → 统计时间 → 日志。"""
    log.section(title)
    if warn:
        log.log(warn, "WARN")
    t0 = time.time()
    extractor = make_extractor(cfg)
    n = extractor.run(str(cfg["exp_dir"]), _make_progress(log, label), stop_check=lambda: STOP.requested)
    secs = time.time() - t0
    log.log(f"{label}提取完成：{n} 条，用时 {_human_dur(secs)}（{n / max(secs, 1e-6):.1f} 条/s）")
    return n


def step_f0(log: TrainLogger, cfg: dict):
    from rvc.train.extract_f0 import TrainF0Extractor
    _run_extraction_step(
        log, cfg,
        title="步骤 2/4 提取 F0（RMVPE）",
        label="F0",
        make_extractor=lambda c: TrainF0Extractor(c["device"], c["fp16"]),
    )


def step_feature(log: TrainLogger, cfg: dict):
    from rvc.train.extract_feature import HuBERTExtractor
    hubert = cfg.get("hubert", "chinese")
    _run_extraction_step(
        log, cfg,
        title=f"步骤 3/4 提取 HuBERT 特征（{hubert}）",
        label="特征",
        make_extractor=lambda c: HuBERTExtractor(c["device"], c["fp16"], hubert=hubert),
        warn="若实验目录之前用另一种特征器提取过特征，请先删除 3_feature768 再重跑（已存在的特征文件会被跳过）",
    )


def _features_ready(exp_dir: Path) -> bool:
    """检查 F0 与 HuBERT 特征是否完整：每个音频文件都有对应特征。"""
    gt_wavs = sorted(exp_dir.glob("0_gt_wavs/*.wav"))
    if not gt_wavs:
        return False
    stems = {p.stem for p in gt_wavs}
    f0_stems = {p.stem for p in exp_dir.glob("2a_f0/*.npy")}
    f0nsf_stems = {p.stem for p in exp_dir.glob("2b-f0nsf/*.npy")}
    feat_stems = {p.stem for p in exp_dir.glob("3_feature768/*.npy")}
    return stems.issubset(f0_stems) and stems.issubset(f0nsf_stems) and stems.issubset(feat_stems)


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
    trainer_mod.WEIGHTS_DIR = model_dir
    log.log(f"导出模型目录已重定向到: {model_dir}")

    # 从头训练时清掉同名旧模型
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
        if info["batch"] == 1:
            _reset_epoch()
        s = state["sum"]
        for key, field in (("d", "loss_d"), ("g", "loss_g"), ("mel", "loss_mel"), ("kl", "loss_kl"), ("fm", "loss_fm")):
            s[key] += info[field]
        state["count"] += 1

    def on_epoch(epoch, total):
        s, n = state["sum"], max(state["count"], 1)
        secs = time.time() - state["t_epoch"]
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
