"""RVC 云 GPU 训练向导主入口。

用法:
    python -m autodl
    python autodl_train.py  (薄包装，等价)
"""
import sys
import warnings

from autodl import (
    DATA_ROOT,
    EXIT_ENV,
    EXIT_OK,
    EXIT_RUNTIME,
    Cancelled,
    EnvFatal,
)
from autodl.envcheck import _probe_assets, _probe_disk, _probe_ffmpeg, _probe_gpu, _probe_packages, _probe_platform
from autodl.logger import TrainLogger
from autodl.prompts import _human_size, ask_yes
from autodl.steps import STOP, _features_ready, _install_signal_handlers, step_f0, step_feature, step_preprocess, step_train
from autodl.wizard import print_summary, run_wizard


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
