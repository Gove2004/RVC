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


def prune_keep_latest(dir_path, pattern: str, keep: int, epoch_of=checkpoint_epoch, on_delete=None) -> list:
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
        path.unlink()
        deleted.append(path)
        if on_delete:
            on_delete(path)
    return deleted


def load_train_json(sr: int):
    path = train_config_path(sr)
    return json.loads(path.read_text(encoding="utf-8"))


def export_model(state_dict, sr: int, config: dict, epoch: int, output_path: str):
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
    # 安全前提：仅检视用户提供的模型文件，来源可信，故 weights_only=False。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    lines = []
    archive = _zip_archive_name(path)
    name = archive if archive else str(ckpt.get("info", Path(path).stem))
    sr = ckpt.get("sr", "unknown")
    version = ckpt.get("version", "unknown")
    f0 = ckpt.get("f0", 1)
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    lines.append(f"真名/模型信息: {name}")
    model_info = str(ckpt.get("info", "")).strip()
    if model_info:
        lines.append(f"Info: {model_info}")
    lines.append(f"文件大小: {file_size_mb:.1f} MB")
    lines.append(f"采样率: {sr}")
    lines.append(f"版本: {version}")
    lines.append(f"F0 支持: {'是' if f0 == 1 else '否'}")
    return "\n".join(lines)


def _rewrite_archive_prefix(src_path: str, new_prefix: str) -> bool:
    old = _zip_archive_name(src_path)
    if old is None or old == new_prefix:
        return False
    old_prefix = old + "/"
    tmp_path = src_path + ".tmpr"
    with zipfile.ZipFile(src_path) as zin:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.startswith(old_prefix):
                    new_fn = new_prefix + item.filename[len(old):]
                else:
                    new_fn = item.filename
                zi = zipfile.ZipInfo(new_fn, date_time=item.date_time)
                zi.compress_type = item.compress_type
                zi.external_attr = item.external_attr
                zi.internal_attr = item.internal_attr
                zout.writestr(zi, data)
    os.replace(tmp_path, src_path)
    return True


def change_archive_name(path: str, new_name: str) -> str:
    new_name = re.sub(r"\.pth$", "", new_name.strip(), flags=re.I).strip()
    if not new_name:
        raise ValueError("新原名不能为空")
    old_name = _zip_archive_name(path)
    if old_name is None:
        raise ValueError("该文件不是 zip 打包的模型，无法修改原名")
    if old_name == new_name:
        return new_name
    _rewrite_archive_prefix(path, new_name)
    return new_name


def change_model_info(path: str, model_info: str) -> None:
    # 安全前提：仅改写用户提供的模型文件，来源可信，故 weights_only=False。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["info"] = re.sub(r"\.pth$", "", model_info, flags=re.I).strip()
    archive = _zip_archive_name(path)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    if archive:
        _rewrite_archive_prefix(tmp_path, archive)
    os.replace(tmp_path, path)


def build_model_config(sr: int, config: dict):
    mel_config = config["data"]
    model = config["model"]
    return [
        mel_config["filter_length"] // 2 + 1,
        32,
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
