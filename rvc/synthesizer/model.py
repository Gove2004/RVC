"""合成器模型 — 统一基类 + v2 768 维变体（带 F0 / 无 F0）"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

import torch
from torch import nn

from rvc.nn import commons
from rvc.synthesizer.encoder import TextEncoder, PosteriorEncoder
from rvc.synthesizer.decoder import Generator, GeneratorNSF
from rvc.synthesizer.flow import ResidualCouplingBlock

sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class _SynthesizerTrnMsBase(nn.Module):
    """统一的 Synthesizer 基类 — 通过 use_f0 参数控制是否使用 F0。"""

    def __init__(
        self,
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
        **kwargs
    ):
        super(_SynthesizerTrnMsBase, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr.get(sr, 48000)
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
        self.use_f0 = use_f0

        # Text Encoder（256 维输入，后续子类会替换为 768 维）
        self.enc_p = TextEncoder(
            256,
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
                is_half=kwargs.get("is_half", True),
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
            "gin_channels: %d, spk_embed_dim: %d, use_f0: %s",
            gin_channels, self.spk_embed_dim, use_f0
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: torch.Tensor,
    ):
        """训练时的前向传播（仅带 F0 模型支持）。"""
        if not self.use_f0:
            raise NotImplementedError("Training forward only supported for F0 models")
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor],
        nsff0: Optional[torch.Tensor],
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        """推理 — 统一接口，根据 use_f0 自动处理 pitch/nsff0 参数。"""
        g = self.emb_g(sid).unsqueeze(-1)

        if skip_head is not None and return_length is not None:
            assert isinstance(skip_head, torch.Tensor)
            assert isinstance(return_length, torch.Tensor)
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = torch.clamp(skip_head - 24, min=0)
            dec_head = head - int(flow_head.item())
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head : head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decoder — 根据 use_f0 传递参数
        if self.use_f0:
            o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        else:
            o = self.dec(z * x_mask, g=g, n_res=return_length2)

        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMsNSFsid(_SynthesizerTrnMsBase):
    """v2 768 维 HuBERT + NSF（带 F0）"""

    def __init__(
        self,
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
        **kwargs
    ):
        super(SynthesizerTrnMsNSFsid, self).__init__(
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
            **kwargs
        )
        # 替换为 768 维 TextEncoder
        del self.enc_p
        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=True,
        )


class SynthesizerTrnMsNSFsid_nono(_SynthesizerTrnMsBase):
    """v2 768 维 HuBERT（无 F0）"""

    def __init__(
        self,
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
        sr=None,
        **kwargs
    ):
        super(SynthesizerTrnMsNSFsid_nono, self).__init__(
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
            **kwargs
        )
        # 替换为 768 维 TextEncoder（无 F0）
        del self.enc_p
        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
