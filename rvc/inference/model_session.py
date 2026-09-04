"""推理模型会话加载。"""
import logging
import os
from dataclasses import dataclass

from rvc.inference.model_loader import SynthesizerLoader
from rvc.models.hubert import load_hubert
from rvc.tools.cuda_graph import clear_cuda_graph_cache

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    """一个模型文件的完整推理会话（HuBERT/Synthesizer 均为共享缓存实例）。"""
    hubert: object
    synthesizer: object
    target_sr: int
    use_f0: int


def load_model_session(config, pth_path: str, inference_cache,
                       hubert_variant: str = "base") -> ModelSession:
    logger.info("加载 %s", os.path.basename(pth_path))

    hubert = load_hubert(config, inference_cache, variant=hubert_variant)
    loader = SynthesizerLoader(config, inference_cache)
    synth = loader.load(pth_path)
    synthesizer = synth["synthesizer"]

    # 所有 Synthesizer 变体都继承 _SynthesizerTrnMsBase.remove_weight_norm（非死分支）；
    # 仅保留异常兜底（权重可能已是移除状态，二次移除会抛 ValueError）
    try:
        synthesizer.remove_weight_norm()
    except Exception as e:
        logger.warning("移除 weight_norm 失败: %s", e)

    # 清除旧模型的 CUDA Graph 缓存（避免形状/设备不匹配）
    clear_cuda_graph_cache(synthesizer)
    clear_cuda_graph_cache(hubert)

    return ModelSession(
        hubert=hubert,
        synthesizer=synthesizer,
        target_sr=synth["target_sr"],
        use_f0=synth["use_f0"],
    )
