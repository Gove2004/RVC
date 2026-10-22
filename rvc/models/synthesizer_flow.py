"""流模型 — ResidualCouplingBlock（可逆耦合流）"""
from typing import Optional

from torch import Tensor, nn

from rvc.nn import modules


class ResidualCouplingBlock(nn.Module):
    """可逆耦合流：训练正向编 z→z_p，推理 reverse=True 解 z_p→z。"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        g: Optional[Tensor] = None,
        reverse: bool = False,
    ) -> Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self) -> None:
        for i in range(self.n_flows):
            flow = self.flows[i * 2]
            assert isinstance(flow, modules.ResidualCouplingLayer)
            flow.remove_weight_norm()
