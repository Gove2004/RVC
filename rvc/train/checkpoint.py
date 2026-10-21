import glob
import json
import logging
import os
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch

from rvc.runtime import train_config_path

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, learning_rate: float, epoch: int, path: str):
    """保存训练 checkpoint（{model, iteration, optimizer, learning_rate}），先写 .tmp 再原子替换。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "iteration": epoch,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "learning_rate": learning_rate,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def save_checkpoint_pair(g_model, g_opt, d_model, d_opt, learning_rate: float,
                         epoch: int, g_path: str, d_path: str):
    """成对保存 G/D checkpoint。

    先写完两个 .tmp 再连续 os.replace——把「G 已落、D 未落」的中断窗口从
    一次 torch.save（数十秒）压缩到两次 replace（微秒）。若中断仍恰好落在
    替换之间，会留下单边孤儿文件，由 prune_orphan_checkpoints 在下次训练前清理。
    """
    save_checkpoint(g_model, g_opt, learning_rate, epoch, g_path + ".pairtmp")
    save_checkpoint(d_model, d_opt, learning_rate, epoch, d_path + ".pairtmp")
    os.replace(g_path + ".pairtmp", g_path)
    os.replace(d_path + ".pairtmp", d_path)


def load_checkpoint(path: str, model, optimizer=None):
    """加载训练 checkpoint（{model, iteration, optimizer, learning_rate}）。

    键匹配语义：形状不一致或缺键的权重被跳过（与上游 strict=False 行为
    一致），但跳过会打 warning——旧版零提示，续训结构不匹配时静默产生
    半随机模型，排查困难。
    """
    # 安全前提：训练 checkpoint 由本流程 torch.save 产出，来源可信，故 weights_only=False。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_state = checkpoint["model"]
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in saved_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    if skipped:
        logger.warning("checkpoint %s 有 %d 个权重未匹配被跳过: %s", path, len(skipped), ", ".join(skipped[:8]))
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["learning_rate"], checkpoint["iteration"]


CHECKPOINT_DIR_NAME = "4_checkpoints"


def checkpoints_dir(exp_dir: str | Path) -> Path:
    return Path(exp_dir) / CHECKPOINT_DIR_NAME


def _epoch_extractor(pattern: str):
    regex = re.compile(pattern)

    def extract(path):
        match = regex.search(str(path))
        return int(match.group(1)) if match else -1

    return extract


checkpoint_epoch = _epoch_extractor(r"_(\d+)\.pth$")
exported_epoch = _epoch_extractor(r"_e(\d+)\.pth$")


def latest_checkpoint_path(dir_path: str, prefix: str):
    paths = glob.glob(os.path.join(dir_path, f"{prefix}_*.pth"))
    if not paths:
        return None
    return max(paths, key=checkpoint_epoch)


def prune_keep_latest(dir_path, pattern: str, keep: int, epoch_of=checkpoint_epoch) -> list:
    """按 epoch 序号淘汰旧文件，只保留最新 keep 组；返回被删除的路径列表。"""
    if keep < 1:
        return []
    directory = Path(dir_path)
    if not directory.is_dir():
        return []
    files = [p for p in directory.glob(pattern) if epoch_of(p) >= 0]
    files.sort(key=epoch_of)
    removed = files[:-keep]
    deleted = []
    for path in removed:
        try:
            path.unlink()
        except OSError:
            # Windows 下文件可能被杀毒/索引器短暂占用；淘汰失败只降级不崩训练
            logger.warning("checkpoint 淘汰失败（跳过）: %s", path)
            continue
        deleted.append(path)
    return deleted


def prune_orphan_checkpoints(exp_dir: str | Path) -> int:
    """删除只有 G 或只有 D 的孤儿 checkpoint（保存中断残留），返回删除数。

    孤儿会同时污染淘汰逻辑（新训低 epoch 被删、陈旧孤儿永存）与续训起点
    （_detect_ckpt_epoch 按 min(G, D) 取 epoch），必须在检测前清理。
    """
    directory = checkpoints_dir(Path(exp_dir))
    if not directory.is_dir():
        return 0

    def _by_epoch(prefix: str) -> dict:
        return {
            checkpoint_epoch(p): p
            for p in directory.glob(f"{prefix}_*.pth") if checkpoint_epoch(p) >= 0
        }

    g, d = _by_epoch("G"), _by_epoch("D")
    removed = 0
    for e in set(g) - set(d):
        g[e].unlink(missing_ok=True)
        removed += 1
    for e in set(d) - set(g):
        d[e].unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.warning("清理孤儿 checkpoint（G/D 只剩单边的中断残留）: %d 个", removed)
    return removed


def load_train_json(sr: int):
    path = train_config_path(sr)
    return json.loads(path.read_text(encoding="utf-8"))


def export_model(state_dict, sr: int, config: dict, epoch: int, output_path: str):
    """导出推理可用模型：剥离 enc_q、权重转 fp16，附带 config/sr/version 元数据。"""
    weights = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("enc_q"):
            continue
        weights[key] = value.detach().cpu().half()
    model_config = build_model_config(sr, config)
    name = re.sub(r"_e\d+$", "", Path(output_path).stem)
    ckpt = {
        "weight": weights,
        "config": model_config,
        "info": name,
        "sr": f"{sr // 1000}k",
        "f0": 1,
        "version": "v2",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, output_path)


def merge_models(path_a: str, path_b: str, ratio: float, output_path: str):
    """按 ratio 加权合并两个同架构模型（config 不同则拒绝）；独有权重原样保留。"""
    # 安全前提：两个待合并模型均由用户提供或本流程导出，来源可信，故 weights_only=False。
    ckpt_a = torch.load(path_a, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(path_b, map_location="cpu", weights_only=False)
    if ckpt_a["config"] != ckpt_b["config"]:
        raise ValueError("两个模型架构不一致（config 不同），无法合并")
    merged = OrderedDict()
    for key in ckpt_a["weight"]:
        if key in ckpt_b["weight"]:
            merged[key] = (ckpt_a["weight"][key].float() * ratio
                           + ckpt_b["weight"][key].float() * (1 - ratio)).half()
        else:
            merged[key] = ckpt_a["weight"][key]
    for key in ckpt_b["weight"]:
        if key not in merged:
            merged[key] = ckpt_b["weight"][key]
    ckpt = {
        "weight": merged,
        "config": ckpt_a["config"],
        "info": Path(output_path).stem,
        "sr": ckpt_a["sr"],
        "f0": ckpt_a.get("f0", 1),
        "version": ckpt_a.get("version", "v2"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, output_path)


def _zip_archive_name(path: str):
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("data.pkl"):
                return n.split("/")[0]
    return None


def inspect_model(path: str) -> str:
    """检视模型：返回多行摘要（模型信息 / zip 原名 / 大小 / 采样率 / 版本 / F0）。"""
    # 安全前提：仅检视用户提供的模型文件，来源可信，故 weights_only=False。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    lines = []
    archive = _zip_archive_name(path)
    # 首行优先展示 info；info 为空时回退 zip 原名 / 文件 stem
    model_info = str(ckpt.get("info", "")).strip()
    name = model_info if model_info else (archive if archive else Path(path).stem)
    lines.append(f"模型信息: {name}")
    # zip 归档原名独立成行——info 为空时与首行可能重复，属可接受冗余，不得省略。
    # archive=None（非 zip 打包）时该行如实显示 None。
    lines.append(f"zip 原名: {archive}")
    sr = ckpt.get("sr", "unknown")
    version = ckpt.get("version", "unknown")
    f0 = ckpt.get("f0", 1)
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    lines.append(f"文件大小: {file_size_mb:.1f} MB")
    lines.append(f"采样率: {sr}")
    lines.append(f"版本: {version}")
    lines.append(f"F0 支持: {'是' if f0 == 1 else '否'}")
    return "\n".join(lines)


# 导出 config 的 segment_size（帧）：与上游 RVC 导出 config 的固定值一致。
# 推理端不使用该值（实时按 block_time 动态计算），仅为保持 config 结构完整可加载。
_MODEL_CONFIG_SEGMENT_SIZE = 32


def build_model_config(sr: int, config: dict):
    mel_config = config["data"]
    model = config["model"]
    return [
        mel_config["filter_length"] // 2 + 1,
        _MODEL_CONFIG_SEGMENT_SIZE,
        model["inter_channels"],
        model["hidden_channels"],
        model["filter_channels"],
        model["n_heads"],
        model["n_layers"],
        model["kernel_size"],
        model["p_dropout"],
        model["resblock"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        model["spk_embed_dim"],
        model["gin_channels"],
        sr,
    ]
