"""推理模型会话加载。"""
import logging
import os

from rvc.inference.index_retrieval import load_index
from rvc.inference.model_loader import SynthesizerLoader
from rvc.models.hubert import load_hubert

logger = logging.getLogger(__name__)


def load_model_session(config, pth_path: str, index_path: str, index_rate: float, inference_cache) -> dict:
    logger.info("加载 %s", os.path.basename(pth_path))

    hubert = load_hubert(config, inference_cache)
    loader = SynthesizerLoader(config, inference_cache)
    synth = loader.load(pth_path)
    net_g = synth["net_g"]

    try:
        net_g.remove_weight_norm()
    except AttributeError:
        logger.debug("模型没有 weight_norm，跳过移除")
    except Exception as e:
        logger.warning("移除 weight_norm 失败: %s", e)

    index = big_npy = None
    if index_rate > 0:
        index, big_npy = load_index(index_path, inference_cache)

    return {
        "hubert": hubert,
        "net_g": net_g,
        "tgt_sr": synth["tgt_sr"],
        "if_f0": synth["if_f0"],
        "version": synth["version"],
        "index": index,
        "big_npy": big_npy,
    }
