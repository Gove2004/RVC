"""合成器模型 — 统一基类 + v2 变体（带 F0 / 无 F0）。

权重文件（`*.pth`）由 `rvc.pipeline.loader` 装配，本模块只描述网络结构。
"""
import logging
from typing import Final, Optional

import torch
from torch import Tensor, nn

from rvc.models.synthesizer_decoder import Generator, GeneratorNSF
from rvc.models.synthesizer_encoder import PosteriorEncoder, TextEncoder
from rvc.models.synthesizer_flow import ResidualCouplingBlock
from rvc.nn import vits_blocks

logger = logging.getLogger(__name__)

# checkpoint 的 sr 字段可能是字符串别名（"48k"）或整数（48000）
_SR_BY_KEY = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

# flow 侧的左侧上下文补偿帧数：flow 分支（WaveNet 感受野更长）必须比
# decoder 分支多看这么多帧，才能让 dec_head 处切出的隐变量与整段推理一致。
# 即 flow 的起点 head - FLOW_CONTEXT_COMPENSATION_FRAMES，decoder 的起点 head。
FLOW_CONTEXT_COMPENSATION_FRAMES = 24


class _SynthesizerTrnMsBase(nn.Module):
    """统一的 Synthesizer 基类 — 通过 use_f0 参数控制是否使用 F0。"""

    __constants__ = ["use_f0"]

    dec: "GeneratorNSF | Generator"
    flow: "ResidualCouplingBlock"
    enc_q: "PosteriorEncoder"

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | str | None,
        use_f0: bool = True,
        phone_dim: int = 256,
        **kwargs: object
    ) -> None:
        """构造合成器。

        `**kwargs` 只用于**吸收**调用方多余的关键字（如 `is_half`）：
        精度由外部 `.half()` 决定，模型内部不用这个信息。
        """
        super(_SynthesizerTrnMsBase, self).__init__()
        if isinstance(sr, str):
            sr = _SR_BY_KEY.get(sr, 48000)
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.use_f0: Final[bool] = use_f0

        # Text Encoder（HuBERT 特征维度：v2 一律 768；256 是基类默认值，v2 子类传 768）
        self.enc_p = TextEncoder(
            phone_dim,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=use_f0,
        )

        # Decoder — 根据 use_f0 选择
        if use_f0:
            assert sr is not None, "use_f0 模型必须提供 sr"
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )

        # Posterior Encoder
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        # Speaker Embedding
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "  · 配置: gin=%d, spk=%d, f0=%s",
            gin_channels, self.spk_embed_dim, use_f0
        )

    def remove_weight_norm(self) -> None:
        self.dec.remove_weight_norm()
        flow = self.flow
        assert isinstance(flow, ResidualCouplingBlock)
        flow.remove_weight_norm()
        # enc_q 可能在加载时被删除（推理不用，省显存），用 hasattr 防护
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def forward(
        self,
        phone: Tensor,
        phone_lengths: Tensor,
        pitch: Tensor,
        pitchf: Tensor,
        y: Tensor,
        y_lengths: Tensor,
        ds: Tensor,
    ) -> tuple:
        """训练时的前向传播（仅带 F0 模型支持）。"""
        if not self.use_f0:
            raise NotImplementedError("Training forward only supported for F0 models")
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = vits_blocks.rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = vits_blocks.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def _z_sample(self, m_p: Tensor, z_noise: Tensor | None) -> Tensor:
        """z_p 采样噪声：传入预生成固定噪声场（形状须与 m_p 一致）则确定性
        采样，否则逐次随机（randn_like，训练与未注入噪声场时的行为）。"""
        if z_noise is not None and z_noise.shape == m_p.shape:
            return z_noise
        return torch.randn_like(m_p)

    def infer(
        self,
        phone: Tensor,
        phone_lengths: Tensor,
        pitch: Optional[Tensor],
        nsff0: Optional[Tensor],
        sid: Tensor,
        skip_head: int = 0,
        return_length: int = 0,
        return_length2: int | Tensor | None = None,
        z_noise: Tensor | None = None,
    ) -> tuple:
        """推理 — 统一接口，根据 use_f0 自动处理 pitch/nsff0 参数。

        f0 契约：nsff0 必须是 2D (B, L)——SineGen.forward 内部会 unsqueeze(-1)
        升成 3D；传 3D 会在 _f02sine 里报 a(40) vs b(39) 形状错。
        随机性：z_p 默认用 randn_like 采样、SineGen 的 rand_ini 初相随机，
        因此每次调用（含逐块调用）输出都不同，属刻意行为非 bug。
        实时链路传 z_noise（预生成的固定噪声场）时 z_p 采样改为确定性：
        逐块独立采样会让持续/重复内容的渲染咬字随机漂移，固定后内容不变
        则渲染不变。z_noise 形状必须与 m_p 一致，不匹配时自动退回随机采样。
        """
        g = self.emb_g(sid).unsqueeze(-1)

        if skip_head > 0 or return_length > 0:
            head = int(skip_head)
            flow_head = max(head - FLOW_CONTEXT_COMPENSATION_FRAMES, 0)
            dec_head = head - flow_head
            length = int(return_length)
            if self.use_f0:
                if pitch is None:
                    raise ValueError("use_f0 模型要求 pitch，实际为 None")
                m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
            else:
                m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * self._z_sample(m_p, z_noise) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head : head + length]
        else:
            if self.use_f0:
                if pitch is None:
                    raise ValueError("use_f0 模型要求 pitch，实际为 None")
                m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            else:
                m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * self._z_sample(m_p, z_noise) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decoder — 根据 use_f0 传递参数
        if self.use_f0:
            if nsff0 is None:
                raise ValueError("use_f0 模型要求 nsff0，实际为 None")
            o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        else:
            o = self.dec(z * x_mask, g=g, n_res=return_length2)

        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMsNSFsid(_SynthesizerTrnMsBase):
    """v2 768 维 HuBERT + NSF（带 F0）"""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | str,
        **kwargs: object
    ) -> None:
        super().__init__(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            use_f0=True,
            phone_dim=768,
            **kwargs
        )


class SynthesizerTrnMsNSFsid_nono(_SynthesizerTrnMsBase):
    """v2 768 维 HuBERT（无 F0）"""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | str | None = None,
        **kwargs: object
    ) -> None:
        super().__init__(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            use_f0=False,
            phone_dim=768,
            **kwargs
        )
