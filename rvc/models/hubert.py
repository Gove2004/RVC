"""HuBERT 模型加载 — 使用 HuggingFace transformers 加载转换后的 HuBERT。

注意力实现走 transformers 的 sdpa（config._attn_implementation），本项目
未安装 flash_attention_2，不要假设其存在。HuBERT 特征 8ms/帧（hop 160 @ 16k）
是链路的硬成本：extra 上下文时长折算出的帧数直接决定推理耗时下限。

**transformers 是延迟导入的**：它会把 sklearn / torch._dynamo / auto_factory
一起拉起来（实测 ~20s），而这只在「真的要加载特征器」时才需要。放在模块顶层
会让任何 `import rvc.pipeline.pipeline`（CLI、测试、离线初始化、甚至只是读取
配置的代码路径）都白付这笔钱。延迟到 load_hubert 首次调用，代价落在本来就要
等模型加载的地方，其余路径零成本。
"""
import logging
import os
import sys
from typing import Any

import torch
from torch import nn

from rvc.core.errors import FeatureExtractionError
from rvc.runtime.paths import HUBERT_ROOT

logger = logging.getLogger(__name__)

# HuBERT 特征器按子目录管理：assets/hubert/base/（原始 hubert_base）
# 与 assets/hubert/chinese/（腾讯 TencentGameMate/chinese-hubert-base，WenetSpeech 中文）。
# 训练与推理必须用同一特征器；每张模型卡牌/每次训练可独立选择 variant。
# HUBERT_ROOT 统一来自 rvc.runtime.paths（assets/hubert）
HUBERT_VARIANTS = ("base", "chinese")

# 延迟构造出的「HubertModel + final_proj」子类（构造一次后复用，保证类型唯一）
_hubert_model_class = None


def _model_class():
    """返回 HubertModelWithFinalProj 类（首次调用时才 import transformers）。"""
    global _hubert_model_class
    if _hubert_model_class is None:
        # 关闭 transformers 加载权重时的 tqdm 进度条（与统一日志格式冲突）
        from transformers import HubertModel
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()

        class HubertModelWithFinalProj(HubertModel):
            """Transformers HuBERT 加 RVC 需要的 final_proj 层。

            forward 直接可调用（transformers 约定，return_dict=True 默认返回
            BaseModelOutput），特征提取层用 outputs.last_hidden_state 取值，
            不需要包一层适配。final_proj 是 RVC 导出模型的固定结构。
            """

            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

        _hubert_model_class = HubertModelWithFinalProj
    return _hubert_model_class


def hubert_path(variant: str = "chinese") -> str:
    """返回指定 HuBERT 变体的模型目录（不校验存在性，由 load_hubert 统一报错）。"""
    return str(HUBERT_ROOT / variant)


def load_hubert(config: Any, variant: str = "chinese") -> Any:
    """加载 HuBERT 特征器（纯加载，不做缓存）。

    缓存策略属于调用方（pipeline 组装层 / 训练侧各自的加载上下文），
    模型层只负责把权重装到设备上。
    """
    if variant not in HUBERT_VARIANTS:
        raise FeatureExtractionError(f"未知 HuBERT 变体: {variant!r}（可选 {HUBERT_VARIANTS}）")
    model_path = hubert_path(variant)
    missing = [f for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin")
               if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        raise FeatureExtractionError(
            f"HuBERT 权重缺失: {model_path}/{missing[0]}（{variant} 特征器未就位，"
            f"请确认 assets/hubert/{variant}/ 三件套完整）"
        )

    dtype = torch.float16 if config.is_half else torch.float32
    logger.info("  · HuBERT（%s, variant=%s）", dtype, variant)

    # GUI 无控制台环境（pythonw / 打包 exe）下 sys.stdout / sys.stderr 为 None，
    # transformers 打印 LOAD REPORT（chinese 有 MISSING 键必触发）会调 sys.stdout.isatty() 崩溃。
    # 保存原始值，from_pretrained 期间临时替换为 devnull，完成后恢复并关闭。
    _stdout, _stderr = sys.stdout, sys.stderr
    try:
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        hubert_model = _model_class().from_pretrained(
            model_path,
            local_files_only=True,
        ).to(device=config.device, dtype=dtype).eval()
    finally:
        if sys.stdout is not _stdout:
            sys.stdout.close()
            sys.stdout = _stdout
        if sys.stderr is not _stderr:
            sys.stderr.close()
            sys.stderr = _stderr

    return hubert_model
