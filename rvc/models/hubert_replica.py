"""HuBERT 纯 torch 复刻 — 推理/训练特征提取共用，不依赖 transformers。

为什么在仓库内复刻
------------------
`from transformers import HubertModel` 会连带拉起 torch._dynamo（经 masking_utils）
与 sklearn（经 generation.candidate_generator），实测导入耗时数秒（热，首次含
dynamo/sklearn 初始化）乃至 70s（冷，主要是杀软首扫数千个 .py/.pyd），并常驻
约 160MB 私有内存（USS）。而 HuBERT 在本项目里**只作冻结特征器**（推理与训练侧都在 eval/inference_mode 下取
last_hidden_state），用不到 transformers 的训练/生成/自动工厂 machinery。复刻后
推理路径不再 import transformers，启动最大变数与这块内存一并消除。

等价性契约
----------
逐字复刻 transformers `modeling_hubert.py` 的**推理前向**算子序列（eval、
attention_mask=None、do_stable_layer_norm=False）：
  conv 栈（首层 GroupNorm(C,C)+gelu，其余 conv+gelu，bias=False）
  → transpose → LayerNorm(512)+Linear(512→768)
  → pos_conv（weight_norm dim=2，k128 pad64 groups16，去末帧）+gelu → 残差+LayerNorm
  → 12×(sdpa 注意力 + 残差 + LayerNorm + FFN(768→3072 gelu→768) + 残差 + LayerNorm)
模块命名与 transformers 完全一致，pytorch_model.bin 可直装。注意力用
F.scaled_dot_product_attention（与 transformers 默认 sdpa 同一后端），保证数值一致。
eval 下 dropout/spec_augment/layerdrop 全为恒等，故复刻不含这些训练态分支。

权重键名兼容
------------
base/ktum 的 pos_conv 用新键名 parametrizations.weight.original0/1，chinese 用旧键名
weight_g/weight_v。nn.utils.parametrizations.weight_norm 的 ParametrizationList 自带
旧键名兼容层（与合成器加载器同一机制），两种键名都能装。chinese 无 final_proj 权重
（推理不用，保持随机初始化即可），masked_spec_embed 三变体都有但前向不用。
"""
import json
import logging
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class _GroupNormConvLayer(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        in_dim = cfg.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = cfg.conv_dim[layer_id]
        self.conv = nn.Conv1d(in_dim, out_dim, cfg.conv_kernel[layer_id],
                              stride=cfg.conv_stride[layer_id], bias=cfg.conv_bias)
        # num_groups == num_channels：逐通道在时间维归一化（与 transformers 一致）
        self.layer_norm = nn.GroupNorm(out_dim, out_dim, affine=True)

    def forward(self, x):
        return F.gelu(self.layer_norm(self.conv(x)))


class _NoNormConvLayer(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        in_dim = cfg.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = cfg.conv_dim[layer_id]
        self.conv = nn.Conv1d(in_dim, out_dim, cfg.conv_kernel[layer_id],
                              stride=cfg.conv_stride[layer_id], bias=cfg.conv_bias)

    def forward(self, x):
        return F.gelu(self.conv(x))


class _FeatureEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # feat_extract_norm="group"：首层 GroupNorm，其余无归一化（三变体均如此）
        layers = [_GroupNormConvLayer(cfg, 0)]
        layers += [_NoNormConvLayer(cfg, i + 1) for i in range(cfg.num_feat_extract_layers - 1)]
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, input_values):
        x = input_values[:, None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class _FeatureProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.conv_dim[-1], eps=cfg.layer_norm_eps)
        self.projection = nn.Linear(cfg.conv_dim[-1], cfg.hidden_size)

    def forward(self, x):
        return self.projection(self.layer_norm(x))


class _PositionalConvEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size,
                              cfg.num_conv_pos_embeddings,
                              padding=cfg.num_conv_pos_embeddings // 2,
                              groups=cfg.num_conv_pos_embedding_groups)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        # k128 为偶数 → SamePadLayer 去末帧
        self._pad_remove = 1 if cfg.num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self._pad_remove:
            x = x[:, :, :-self._pad_remove]
        x = F.gelu(x)
        return x.transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        # 与 transformers 默认 sdpa 同后端：无 mask、eval dropout=0、非因果
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                             dropout_p=0.0, is_causal=False, scale=self.scaling)
        out = out.transpose(1, 2).reshape(b, t, -1).contiguous()
        return self.out_proj(out)


class _FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.intermediate_dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output_dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class _EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = _Attention(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.feed_forward = _FeedForward(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        return self.final_layer_norm(x)


class _Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_conv_embed = _PositionalConvEmbedding(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList([_EncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x):
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class HubertReplica(nn.Module):
    """纯 torch HuBERT（模块命名对齐 transformers，权重直装）。

    forward 返回 SimpleNamespace(last_hidden_state=...)，与 transformers
    BaseModelOutput 的取值方式一致（hubert_forward 取 .last_hidden_state）。
    config 暴露 conv_kernel/conv_stride 供 hubert_frame_count 折算帧数。
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.feature_extractor = _FeatureEncoder(cfg)
        self.feature_projection = _FeatureProjection(cfg)
        self.encoder = _Encoder(cfg)
        # final_proj / masked_spec_embed 在权重里但前向不用：建出来只为直装权重
        self.final_proj = nn.Linear(cfg.hidden_size, cfg.classifier_proj_size)
        self.masked_spec_embed = nn.Parameter(torch.zeros(cfg.hidden_size), requires_grad=False)

    def forward(self, input_values):
        x = self.feature_extractor(input_values).transpose(1, 2)
        x = self.feature_projection(x)
        x = self.encoder(x)
        return SimpleNamespace(last_hidden_state=x)


def load_replica(model_path: str, device: str, is_half: bool) -> HubertReplica:
    """从 assets/hubert/<variant>/ 三件套加载纯 torch HuBERT。"""
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        cfg = SimpleNamespace(**json.load(f))

    model = HubertReplica(cfg)
    state = torch.load(os.path.join(model_path, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    # strict=False：chinese 缺 final_proj（预期缺失，推理不用）；其余键应全装。
    # weight_g/weight_v 旧键名经 ParametrizationList 兼容层映射到 original0/1。
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("final_proj.")]
    unexpected = list(result.unexpected_keys)
    if missing:
        logger.warning("HuBERT 权重缺失键（保持随机初始化，输出可能异常）: %s", missing[:8])
    if unexpected:
        logger.warning("HuBERT 权重含多余键（已忽略）: %s", unexpected[:8])
    del state

    dtype = torch.float16 if is_half else torch.float32
    return model.to(device=device, dtype=dtype).eval()