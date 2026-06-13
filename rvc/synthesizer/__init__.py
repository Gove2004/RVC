"""合成器模块 — RVC 语音合成器的各子组件"""
import torch

from rvc.synthesizer.encoder import PosteriorEncoder, TextEncoder
from rvc.synthesizer.decoder import Generator, GeneratorNSF, SineGen, SourceModuleHnNSF
from rvc.synthesizer.flow import ResidualCouplingBlock
from rvc.synthesizer.model import (
    SynthesizerTrnMsNSFsid,
    SynthesizerTrnMsNSFsid_nono,
    sr2sr,
)


def load_synthesizer_for_jit(pth_path: str, device):
    """加载 Synthesizer 用于 JIT 导出（不移除 enc_q）。

    Args:
        pth_path: .pth 模型路径
        device: 目标设备

    Returns:
        (net_g, cpt): 模型和检查点字典
    """
    cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v2")

    if version == "v1":
        if if_f0 == 1:
            from rvc.synthesizer.model import SynthesizerTrnMs256NSFsid
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            from rvc.synthesizer.model import SynthesizerTrnMs256NSFsid_nono
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    else:  # v2
        if if_f0 == 1:
            net_g = SynthesizerTrnMsNSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMsNSFsid_nono(*cpt["config"])

    # JIT 导出需要保留 enc_q
    # del net_g.enc_q

    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


__all__ = [
    "TextEncoder",
    "PosteriorEncoder",
    "ResidualCouplingBlock",
    "Generator",
    "GeneratorNSF",
    "SineGen",
    "SourceModuleHnNSF",
    "SynthesizerTrnMsNSFsid",
    "SynthesizerTrnMsNSFsid_nono",
    "sr2sr",
    "load_synthesizer_for_jit",
]
