import glob
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch

from configs.config import runtime_train_config_path


def save_checkpoint(model, optimizer, learning_rate: float, epoch: int, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "iteration": epoch,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "learning_rate": learning_rate,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_state = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    matched = {}
    for key, value in saved_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("learning_rate", 1e-4), checkpoint.get("iteration", 0)


def latest_checkpoint_path(dir_path: str, prefix: str):
    paths = glob.glob(os.path.join(dir_path, f"{prefix}_*.pth"))
    if not paths:
        return None

    def epoch_of(path):
        match = re.search(rf"{prefix}_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    return max(paths, key=epoch_of)


def load_train_json(sr: int):
    path = runtime_train_config_path(sr)
    return json.loads(path.read_text(encoding="utf-8"))


def export_model(state_dict, sr: int, config: dict, epoch: int, output_path: str):
    weights = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("enc_q"):
            continue
        weights[key] = value.detach().cpu().half()
    model_config = build_model_config(sr, config)
    ckpt = {
        "weight": weights,
        "config": model_config,
        "info": f"{epoch}epoch",
        "sr": "48k" if sr == 48000 else "32k",
        "f0": 1,
        "version": "v2",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)


def merge_models(path_a: str, path_b: str, ratio: float, output_path: str):
    """合并两个 .pth 推理模型，按比例加权平均权重。ratio 为模型 A 的占比。"""
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
        "info": f"merged A*{ratio:.2f}",
        "sr": ckpt_a["sr"],
        "f0": ckpt_a.get("f0", 1),
        "version": ckpt_a.get("version", "v2"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)



def inspect_model(path: str) -> str:
    """加载 .pth 模型，返回基本信息文本。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    lines = []
    info = ckpt.get("info", "")
    sr = ckpt.get("sr", "unknown")
    version = ckpt.get("version", "unknown")
    f0 = ckpt.get("f0", 1)
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    if info:
        lines.append(f"模型信息: {info}")
    lines.append(f"文件: {Path(path).name}")
    lines.append(f"文件大小: {file_size_mb:.1f} MB")
    lines.append(f"采样率: {sr}")
    lines.append(f"版本: {version}")
    lines.append(f"F0 支持: {'是' if f0 == 1 else '否'}")
    return "\n".join(lines)


def build_model_config(sr: int, config: dict):
    data = config["data"]
    model = config["model"]
    return [
        data["filter_length"] // 2 + 1,
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
