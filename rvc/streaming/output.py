"""实时输出路由 — 主输出写入与副输出分发。"""
import logging
import queue

import numpy as np
import torch

logger = logging.getLogger(__name__)


def write_main_output(chunk: torch.Tensor, outdata: np.ndarray, channels: int,
                      out_pin: torch.Tensor | None = None) -> None:
    """把输出块写入硬件缓冲区。

    out_pin：预分配的 pinned 中转缓冲（与输入侧 in_pin 对称）。提供且足够大时，
    D2H 走 pinned DMA——避免每块分配 pageable 临时内存 + 慢速拷贝；
    长度异常（超过预分配）或未提供时回退 chunk.cpu().numpy()。

    并发安全说明：D2H 拷贝是同步的（non_blocking=False）——声卡回调线程必须
    在本函数返回前拿到数据；pinned 缓冲仅回调线程读写，无跨线程竞争。
    """
    if out_pin is not None and chunk.shape[0] <= out_pin.shape[0]:
        buf = out_pin[: chunk.shape[0]]
        buf.copy_(chunk)  # 同步 D2H（pinned DMA），拷完才返回
        out_chunk = buf.numpy()  # pinned 内存上的零拷贝视图
    else:
        out_chunk = chunk.cpu().numpy()
    if channels == 1:
        outdata[:, 0] = out_chunk
    else:
        outdata[:] = out_chunk[:, None]


def route_secondary_output(outdata: np.ndarray, stream2, out2_q: queue.Queue) -> None:
    """把主输出块复制到副输出队列。

    调用方已判断 enable_out2（外层 if），这里只需要检查 stream2 存在即可。
    副输出流存在但未启用时，out2_callback 读到空队列输出静音，无需在此处理。
    """
    if stream2:
        if out2_q.full():
            try:
                out2_q.get_nowait()
            except queue.Empty:
                pass
        out2_q.put_nowait(outdata.copy())
