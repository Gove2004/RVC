"""训练命令行向导主流程。

用法:
    python app.py --train [数据集目录]
    python -m rvc.train.cli [数据集目录]

Windows 与 Linux 同一套流程：环境体检 → 素材检查 → 参数向导 → 预处理 →
F0 → HuBERT 特征 → 训练 → 汇总。逐步跳过已完成的阶段（manifest 指纹 / 特征完整性）。
"""
import os
import sys
import warnings

from rvc.runtime.paths import PROJECT_ROOT, TRAIN_LOGS_ROOT

from rvc.train.cli import (
    EXIT_ENV,
    EXIT_OK,
    EXIT_RUNTIME,
    LOG_NAME,
    Cancelled,
    EnvFatal,
)
from rvc.train.cli.envcheck import (
    _probe_assets,
    _probe_ffmpeg,
    _probe_gpu,
    _probe_packages,
    _probe_platform,
)
from rvc.train.cli.logger import TrainLogger
from rvc.train.cli.prompts import _human_size, ask_yes
from rvc.train.cli.steps import (
    STOP,
    _features_ready,
    _install_signal_handlers,
    step_f0,
    step_feature,
    step_preprocess,
    step_train,
)
from rvc.train.cli.wizard import print_summary, run_wizard


def main(argv: list[str] | None = None) -> int:
    """argv: 传给向导的参数（第一个非选项参数视为数据集路径）。

    None 时取 sys.argv[1:]（`python -m rvc.train.cli` 路径）。
    """
    args = list(sys.argv[1:] if argv is None else argv)

    # 项目内存在以 cwd 为基准的相对引用，入口统一基准目录（进程级副作用，CLI 入口可接受）
    os.chdir(PROJECT_ROOT)

    # Trainer 里 scheduler.step() 先于 optimizer.step() 的既有写法会每轮刷两条警告，
    # 属于原代码行为，这里只在脚本侧静音，不改源文件
    warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")

    argv_dir = ""
    for arg in args:
        if not arg.startswith("-"):
            argv_dir = arg
            break  # 只取第一个非选项参数（与 docstring 口径一致）

    log = TrainLogger()
    try:
        log.attach(TRAIN_LOGS_ROOT / LOG_NAME)
        log.section("RVC 训练向导")

        log.section("第 1 步 / 环境检查")
        _probe_platform(log)
        if _probe_packages(log):
            return EXIT_ENV
        device, fp16, vram_gb = _probe_gpu(log)
        ffmpeg_exe = _probe_ffmpeg(log)
        log.log(f"最终设备    : device={device}  fp16={fp16}")

        log.section("第 2 步 / 预训练模型检查")
        _probe_assets(log)

        cfg = run_wizard(log, device, fp16, vram_gb, argv_dir, ffmpeg_exe=ffmpeg_exe)

        # 预处理指纹核对放在「配置确认」之前：重新预处理会连同 checkpoint 一起清掉，
        # 若等到确认完才发现要重跑，摘要里的「从 epoch N 续训」就是假的
        from rvc.train.preprocess import manifest_diff_reason

        exp_dir = cfg["exp_dir"]
        reason = manifest_diff_reason(exp_dir, cfg["input_dir"], cfg["sr_hz"], cfg["per"])
        if reason and cfg["ckpt_epoch"]:
            log.log(f"预处理需要重跑（{reason}）→ 已有 checkpoint（epoch {cfg['ckpt_epoch']}）"
                    "会被一并清除，将从第 1 轮重新训练", "WARN")
            cfg["ckpt_epoch"] = 0
        print_summary(log, cfg)

        if not ask_yes("开始训练？（训练中 Ctrl+C = 本轮跑完保存后退出）", default=True):
            log.log("已取消，未开始训练", "WARN")
            return EXIT_OK

        _install_signal_handlers(log)
        exp_dir.mkdir(parents=True, exist_ok=True)

        if reason:
            log.log(f"需要重新预处理：{reason}")
            step_preprocess(log, cfg)
        else:
            log.log(f"切片与素材一致，跳过预处理（要重建请删 {exp_dir / 'manifest.json'}）")

        if not STOP.requested:
            hubert = cfg.get("hubert", "chinese")
            if _features_ready(exp_dir, hubert):
                log.log("F0 与 HuBERT 特征已完整，跳过提取")
            else:
                # 特征目录存在但变体不符（或无记录）→ 整目录删除重提。
                # 不删的话 extract 会跳过已存在文件，新旧特征静默混用。
                feat_dir = exp_dir / "3_feature768"
                variant_file = feat_dir / ".variant"
                if feat_dir.is_dir() and (not variant_file.exists()
                        or variant_file.read_text(encoding="utf-8").strip() != hubert):
                    import shutil

                    shutil.rmtree(feat_dir)
                    log.log(f"3_feature768 由其他特征器提取（或无记录），与本次选择的 {hubert} 不符 → 已删除，将重新提取", "WARN")
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
