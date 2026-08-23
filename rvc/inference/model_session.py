"""推理模型会话加载。"""
import logging
import os

from rvc.inference.index_retrieval import load_index
from rvc.inference.model_loader import SynthesizerLoader
from rvc.models.hubert import load_hubert
from rvc.tools.cuda_graph import clear_cuda_graph_cache

logger = logging.getLogger(__name__)


def load_model_session(config, pth_path: str, index_path: str, index_rate: float, inference_cache) -> dict:
    logger.info("加载 %s", os.path.basename(pth_path))

    hubert = load_hubert(config, inference_cache)
    loader = SynthesizerLoader(config, inference_cache)
    synth = loader.load(pth_path)
    synthesizer = synth["synthesizer"]

    try:
        synthesizer.remove_weight_norm()
    except AttributeError:
        logger.debug("模型没有 weight_norm，跳过移除")
    except Exception as e:
        logger.warning("移除 weight_norm 失败: %s", e)

    # 清除旧模型的 CUDA Graph 缓存（避免形状/设备不匹配）
    clear_cuda_graph_cache(synthesizer)
    hubert_obj = getattr(hubert, "model", hubert)
    clear_cuda_graph_cache(hubert_obj)

    index = index_vectors = None
    if index_rate > 0:
        index, index_vectors = load_index(index_path, inference_cache)

    return {
        "hubert": hubert,
        "synthesizer": synthesizer,
        "target_sr": synth["target_sr"],
        "use_f0": synth["use_f0"],
        "index": index,
        "index_vectors": index_vectors,
    }
