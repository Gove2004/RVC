"""TorchScript JIT 编译工具 — 加速推理性能（10-30% 提升）。

用法：
    # 导出 JIT 模型
    cpt = synthesizer_jit_export(
        model_path="path/to/model.pth",
        device=torch.device("cuda:0"),
        is_half=True
    )

    # 加载 JIT 模型
    cpt = load("path/to/model.half.jit")
    net_g = torch.jit.load(BytesIO(cpt["model"]), map_location=device)
    net_g.infer = net_g.forward
"""
import pickle
from io import BytesIO
from collections import OrderedDict

import torch


def export(
    model: torch.nn.Module,
    mode: str = "script",
    inputs: dict = None,
    device=torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    """导出模型为 TorchScript JIT 格式。

    Args:
        model: PyTorch 模型
        mode: "script" | "trace"，默认 script（更通用）
        inputs: trace 模式需要的示例输入
        device: 目标设备
        is_half: 是否使用半精度

    Returns:
        dict: {"model": bytes, "is_half": bool}
    """
    model = model.half() if is_half else model.float()
    model.eval()

    if mode == "trace":
        assert inputs is not None, "trace 模式需要提供 inputs"
        model_jit = torch.jit.trace(model, example_kwarg_inputs=inputs)
    elif mode == "script":
        model_jit = torch.jit.script(model)
    else:
        raise ValueError(f"未知 mode: {mode}")

    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()

    buffer = BytesIO()
    torch.jit.save(model_jit, buffer)
    del model_jit

    cpt = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def load(path: str) -> dict:
    """加载 JIT 缓存文件。

    Args:
        path: .jit 文件路径

    Returns:
        dict: 包含 "model" (bytes)、"device" (str)、"config"、"f0"、"version" 等
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save(ckpt: dict, save_path: str):
    """保存 JIT 缓存文件。

    Args:
        ckpt: JIT 模型字典
        save_path: 保存路径
    """
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)


def synthesizer_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
) -> dict:
    """导出 Synthesizer 为 JIT 格式。

    Args:
        model_path: .pth 模型路径
        mode: "script" | "trace"
        inputs_path: trace 模式的输入示例路径
        save_path: JIT 文件保存路径（默认为 model_path.half.jit 或 .jit）
        device: 目标设备
        is_half: 是否半精度

    Returns:
        dict: JIT 缓存字典（包含 config、f0、version、model bytes、device）
    """
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"

    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")

    # 加载原始模型
    from rvc.synthesizer import load_synthesizer_for_jit
    model, cpt = load_synthesizer_for_jit(model_path, device)

    # 替换 forward 为 infer（JIT 要求）
    model.forward = model.infer

    inputs = None
    if mode == "trace":
        # 这里需要加载示例输入（暂不实现 trace 模式）
        raise NotImplementedError("trace 模式暂未实现，请使用 script 模式")

    # 导出 JIT
    ckpt = export(model, mode, inputs, device, is_half)

    # 组装完整缓存
    cpt.pop("weight")  # 移除原始权重（JIT 已包含）
    cpt["model"] = ckpt["model"]
    cpt["device"] = str(device)

    save(cpt, save_path)
    return cpt
