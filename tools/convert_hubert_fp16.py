"""把 HuBERT 权重转成半精度（fp16）——把"假的 fp32 精度"去掉。

## 为什么需要这个工具

本项目**全程 fp16**：推理侧 `rvc/runtime/device.py` 把 `is_half` 硬编码为 True，
HuBERT 加载后立刻 `.to(dtype=torch.float16)`。所以权重文件里的 fp32 低位在加载的
第一瞬间就被丢掉，只白白多占一倍磁盘，`torch.load` 期间还多占一倍宿主内存。

官方分发的 `chinese` 权重是 fp32（约 360MB），`base` 是 fp16（约 189MB）。把前者
转成 fp16 后加载结果**逐位相同**，但省一半磁盘。本工具就是做这件事，并证明它无损。

## 用法

    python tools/convert_hubert_fp16.py            # 处理全部变体
    python tools/convert_hubert_fp16.py chinese    # 只处理指定变体
    python tools/convert_hubert_fp16.py --dry-run  # 只报告不落盘

## 安全策略

原文件在被覆盖前会改名为 `pytorch_model.fp32.bak`，只有全部验证通过才删除。
任何一道关卡失败都保留备份并回滚，绝不留下半成品。

## 两道验证关卡

A. 逐张量：`原始值.half()` 必须与新文件逐位相等（转换目标精度内无损）。
B. 端到端：两份权重各自加载成 fp16 模型，对同一段确定性信号取特征，必须逐位相等。

关卡 B 才是决定性的——它证明的是"实际拿到的特征不变"，而不只是"文件格式变了"。
用合成信号而非外部音频，保证本工具在任何机器上结果可复现。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from rvc.core.config import HUBERT_VARIANTS  # noqa: E402
from rvc.models.hubert import _model_class, hubert_path  # noqa: E402

WEIGHT_NAME = "pytorch_model.bin"
BACKUP_NAME = "pytorch_model.fp32.bak"


def _is_all_half(sd: dict) -> bool:
    tensors = [v for v in sd.values() if torch.is_tensor(v)]
    return bool(tensors) and all(t.dtype == torch.float16 for t in tensors)


def _probe_signal(seconds: float = 2.0, sr: int = 16000) -> torch.Tensor:
    """确定性测试信号：扫频 + 固定种子噪声。

    覆盖高低频与静音段，避免"只测单音"时某些权重分支压根没被激活。
    """
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    # 200Hz → 800Hz 线性扫频
    freq = 200.0 + 600.0 * (t / t[-1])
    phase = 2 * torch.pi * torch.cumsum(freq / sr, dim=0)
    sweep = 0.5 * torch.sin(phase)
    gen = torch.Generator().manual_seed(20261027)
    noise = 0.05 * torch.randn(n, generator=gen)
    return sweep + noise


def _load_dir(dirpath: Path):
    cls = _model_class()
    return cls.from_pretrained(str(dirpath), local_files_only=True) \
        .to(device="cpu", dtype=torch.float16).eval()


def _features(model, wav16: torch.Tensor) -> torch.Tensor:
    from rvc.pipeline.features import hubert_forward

    with torch.inference_mode():
        return hubert_forward(model, wav16.unsqueeze(0).to(torch.float16))


def _temp_dir_with(src_dir: Path, weight_file: Path) -> Path:
    d = Path(tempfile.mkdtemp(prefix="hub_fp16_"))
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(src_dir / name, d / name)
    shutil.copy(weight_file, d / WEIGHT_NAME)
    return d


def convert_variant(variant: str, dry_run: bool) -> bool:
    """处理单个变体，返回是否成功（或无需处理）。"""
    src_dir = Path(hubert_path(variant))
    src = src_dir / WEIGHT_NAME
    print(f"\n=== {variant} ===")
    if not src.exists():
        print(f"  跳过：权重不存在（{src}）")
        return True

    orig_size = src.stat().st_size
    print(f"  当前：{orig_size:,} 字节（{orig_size / 1024 / 1024:.1f} MB）")

    sd = torch.load(src, map_location="cpu", weights_only=True)
    if _is_all_half(sd):
        print("  已是全 fp16，无需转换")
        return True

    n_fp32 = sum(1 for v in sd.values() if torch.is_tensor(v) and v.dtype != torch.float16)
    print(f"  待转换张量：{n_fp32} / {len(sd)}")

    sd16 = {k: (v.half() if torch.is_tensor(v) else v) for k, v in sd.items()}
    tmp = src_dir / (WEIGHT_NAME + ".fp16.tmp")
    torch.save(sd16, tmp)
    new_size = tmp.stat().st_size
    print(f"  转换后：{new_size:,} 字节（{new_size / 1024 / 1024:.1f} MB），"
          f"节省 {(orig_size - new_size) / 1024 / 1024:.1f} MB")

    if dry_run:
        tmp.unlink()
        print("  --dry-run：不落盘，已清理临时文件")
        return True

    # ── 关卡 A ──
    print("  [A] 逐张量核对 ...", flush=True)
    reloaded = torch.load(tmp, map_location="cpu", weights_only=True)
    if set(reloaded) != set(sd):
        print("    ✗ 键集合不一致"); tmp.unlink(); return False
    bad = [k for k in sd
           if torch.is_tensor(sd[k]) and not torch.equal(sd[k].half(), reloaded[k])]
    if bad:
        print(f"    ✗ {len(bad)} 个张量不一致，样例 {bad[:3]}"); tmp.unlink(); return False
    print(f"    ✓ {len(sd)} 个张量全部一致")

    # ── 关卡 B ──
    print("  [B] 端到端特征比对 ...", flush=True)
    wav16 = _probe_signal()
    tmp_dir = _temp_dir_with(src_dir, tmp)
    try:
        m_orig = _load_dir(src_dir)
        m_new = _load_dir(tmp_dir)
        f_orig = _features(m_orig, wav16)
        f_new = _features(m_new, wav16)
        del m_orig, m_new
        if f_orig.shape != f_new.shape:
            print(f"    ✗ 形状不同 {tuple(f_orig.shape)} vs {tuple(f_new.shape)}")
            shutil.rmtree(tmp_dir, ignore_errors=True); tmp.unlink(); return False
        if not torch.equal(f_orig, f_new):
            diff = (f_orig.float() - f_new.float()).abs().max().item()
            print(f"    ✗ 特征不一致，最大绝对差 {diff:.3e}")
            shutil.rmtree(tmp_dir, ignore_errors=True); tmp.unlink(); return False
        print(f"    ✓ 特征逐位相等，形状 {tuple(f_orig.shape)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 替换（备份到最后） ──
    bak = src_dir / BACKUP_NAME
    shutil.move(str(src), str(bak))
    shutil.move(str(tmp), str(src))

    print("  [复核] 替换后文件 ...", flush=True)
    m_check = _load_dir(src_dir)
    f_check = _features(m_check, wav16)
    del m_check
    if not torch.equal(f_orig, f_check):
        print("    ✗ 替换后特征不一致 → 回滚")
        shutil.move(str(src), str(tmp))
        shutil.move(str(bak), str(src))
        tmp.unlink(missing_ok=True)
        return False
    print("    ✓ 一致")

    bak.unlink()
    final_size = src.stat().st_size
    print(f"  完成：{orig_size:,} → {final_size:,} 字节"
          f"（节省 {(orig_size - final_size) / 1024 / 1024:.1f} MB），备份已删除")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把 HuBERT 权重转成 fp16（本项目全程半精度，fp32 存盘只是浪费）")
    ap.add_argument("variants", nargs="*", default=None,
                    help=f"要处理的变体，默认全部（可选 {', '.join(HUBERT_VARIANTS)}）")
    ap.add_argument("--dry-run", action="store_true", help="只报告不落盘")
    args = ap.parse_args(argv)

    targets = args.variants or list(HUBERT_VARIANTS)
    unknown = [v for v in targets if v not in HUBERT_VARIANTS]
    if unknown:
        ap.error(f"未知变体 {unknown}（可选 {list(HUBERT_VARIANTS)}）")

    ok = all(convert_variant(v, args.dry_run) for v in targets)
    print("\n结论：", "全部处理成功" if ok else "有失败项，原文件已回滚")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
