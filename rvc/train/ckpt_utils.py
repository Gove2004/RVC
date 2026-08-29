import glob
import json
import os
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch

from rvc.runtime import train_config_path


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
    path = train_config_path(sr)
    return json.loads(path.read_text(encoding="utf-8"))


def export_model(state_dict, sr: int, config: dict, epoch: int, output_path: str):
    weights = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("enc_q"):
            continue
        weights[key] = value.detach().cpu().half()
    model_config = build_model_config(sr, config)
    # info = 实验名（文件名去 _e{epoch} 与 .pth）。与原版「改名」/照妖镜对齐：
    # 真名即文件名，不附加任何自定义后缀，也不新增参数。
    name = re.sub(r"_e\d+$", "", Path(output_path).stem)
    ckpt = {
        "weight": weights,
        "config": model_config,
        "info": name,
        "sr": f"{sr // 1000}k",  # 40k/48k，此前硬编码 48k 导致 40k 模型标注错误
        "f0": 1,
        "version": "v2",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, output_path)


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
        # info = 合并名（即输出文件名，去 .pth），与原版「改名」/照妖镜对齐：
        # 真名即文件名，不附加任何自定义来源/比例后缀。
        "info": Path(output_path).stem,
        "sr": ckpt_a["sr"],
        "f0": ckpt_a.get("f0", 1),
        "version": ckpt_a.get("version", "v2"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, output_path)



def _coerce(v) -> str:
    """把可能的非纯 str 形态（numpy.str_ / bytes / 0 维张量等）归一为 str。"""
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if isinstance(v, str):
        return v
    if isinstance(v, torch.Tensor) and v.dim() == 0:
        try:
            return str(v.item())
        except Exception:
            return ""
    try:
        return str(v)
    except Exception:
        return ""


def _extract_info(ckpt: dict, path: str):
    """兜底提取真名（非 zip 老格式 / zip 前缀读不到时）：优先 info，再文件名。"""
    info = _coerce(ckpt.get("info")).strip()
    if info:
        return info, False
    model = ckpt.get("model")
    if isinstance(model, dict):
        s2 = _coerce(model.get("info")).strip()
        if s2:
            return s2, False
    return Path(path).stem, True


def _zip_archive_name(path: str):
    """torch.save 的 zip 序列化把模型存为 <archive>/data.pkl、<archive>/data/0…，
    <archive> 就是保存时的文件名（改名外层 .pth 不会改变它）。这正是『改名照妖镜』
    揭示原名（如 BJX2）的来源——真名焊在 zip 内部路径前缀里。返回该 archive 名，
    读不到（非 zip / 老 pickle 格式）返回 None。"""
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith("data.pkl"):
                    return n.split("/")[0]
    except Exception:
        return None
    return None


def inspect_model(path: str) -> str:
    """加载 .pth 模型，返回基本信息文本。真名优先取 zip 内部 archive 名
    （与照妖镜对齐，抗改名）；读不到再回退 info / 文件名。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    lines = []
    # 真名 = zip 内部 archive 名（照妖镜来源）；取不到才回退 info/文件名
    archive = _zip_archive_name(path)
    name = archive if archive else ""
    if not name:
        name, _ = _extract_info(ckpt, path)
    sr = ckpt.get("sr", "unknown")
    version = ckpt.get("version", "unknown")
    f0 = ckpt.get("f0", 1)
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    lines.append(f"真名/模型信息: {name}")
    # info 与 zip 原名是两件独立的事，分开展示
    info_val = _coerce(ckpt.get("info")).strip()
    if info_val:
        lines.append(f"Info: {info_val}")
    lines.append(f"文件大小: {file_size_mb:.1f} MB")
    lines.append(f"采样率: {sr}")
    lines.append(f"版本: {version}")
    lines.append(f"F0 支持: {'是' if f0 == 1 else '否'}")
    return "\n".join(lines)


def _rewrite_archive_prefix(src_path: str, new_prefix: str) -> bool:
    """把 src_path 这个 zip 模型全部条目前缀改为 new_prefix，原地写回（tmp+replace）。
    纯重命名条目名、不动张量数据；张量存储键在 data.pkl 内是相对路径（如 data/0），
    不含前缀，所以 torch.load 改名后仍可正常加载。返回是否实际发生了重命名。"""
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
    """修改 zip 内部 archive 名（即照妖镜显示的原名 / 真名）。

    torch.save 把模型存为 <archive>/data.pkl、<archive>/data/0…，<archive> 是
    保存时的文件名，改名外层 .pth 不会改变它（这就是『改名照妖镜』的原理）。
    本函数纯重命名 zip 条目名的前缀，不动任何张量数据，写回原文件。"""
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


def change_info(path: str, info: str) -> None:
    """修改模型的 info 字段（工具内可读元信息），原子写回原文件。

    关键：torch.save 每次都会生成随机 zip 前缀，会冲掉原本的 archive 名（原名 /
    照妖镜显示名）。这里先记下原有前缀，save 完再把它还原回去，使 name 与 info 互不干扰。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["info"] = re.sub(r"\.pth$", "", info, flags=re.I).strip()
    archive = _zip_archive_name(path)  # 记下原有原名
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    if archive:
        _rewrite_archive_prefix(tmp_path, archive)  # 还原原名
    os.replace(tmp_path, path)


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
