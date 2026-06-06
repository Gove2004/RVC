import glob
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch


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
    sr_name = "48k" if sr == 48000 else "32k"
    path = Path("configs") / "v2" / f"{sr_name}.json"
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
