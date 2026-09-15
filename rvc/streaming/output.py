"""实时输出路由 — 主输出写入与副输出分发。"""
import logging
import queue

import numpy as np
import torch

logger = logging.getLogger(__name__)


def write_main_output(chunk: torch.Tensor, outdata: np.ndarray, channels: int) -> None:
    out_chunk = chunk.cpu().numpy()
    if channels == 1:
        outdata[:, 0] = out_chunk
    else:
        outdata[:] = out_chunk[:, None]


def route_secondary_output(outdata: np.ndarray, stream2, out2_q: queue.Queue, enable_out2: bool) -> None:
    """把主输出块复制到副输出队列（仅启用时）。

    副输出流存在但未启用时，out2_callback 读到空队列输出静音，无需在此处理。
    """
    if stream2 and enable_out2:
        if out2_q.full():
            try:
                out2_q.get_nowait()
            except Exception:
                pass
        out2_q.put_nowait(outdata.copy())
