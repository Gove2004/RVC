"""实时输出路由 — 主输出写入与副输出块队列。

两条实时性约束决定了本模块的形状：
- 全部跑在声卡回调线程里，不能阻塞、不能做无界分配；
- 主输出必须在本函数返回前落到硬件缓冲（同步 D2H），副输出则允许丢块。
"""
import queue

import numpy as np
import torch


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


class Out2Queue:
    """副输出块队列 — 固定容量 + 缓冲复用。

    一进一出两个声卡回调线程：主输出回调 push，副输出回调 take_into。
    块不在每次 push 时新分配（旧实现每块 outdata.copy() 一次，1920 帧立体声
    即 15KB/块），而是循环使用队列自己持有的 (容量 + 1) 块：取走拷完立即
    归还，队列满时被丢弃的最旧块同样归还。稳态分配为零，也不随时间增长。

    线程安全：_q 自带锁；_free 只用 list.append / list.pop（CPython 下是原子
    操作），「先查后用」的竞态用 IndexError 兜住，因此不必再引入额外锁——
    回调线程里每多一把锁都是风险。
    """

    def __init__(self, maxsize: int = 10) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._free: list[np.ndarray] = []

    def push(self, block: np.ndarray) -> None:
        """拷入一块空闲缓冲并入队；队列满则丢最旧一块（与旧行为一致）。"""
        buf = self._acquire(block.shape)
        buf[...] = block
        while True:
            try:
                self._q.put_nowait(buf)
                return
            except queue.Full:
                try:
                    self._free.append(self._q.get_nowait())
                except queue.Empty:
                    pass

    def take_into(self, outdata: np.ndarray, frames: int) -> bool:
        """取一块写入设备缓冲并归还缓冲。

        Returns:
            True = 已写入；False = 队列空，调用方应输出静音。
        """
        try:
            buf = self._q.get_nowait()
        except queue.Empty:
            return False
        n = min(frames, buf.shape[0])
        outdata[:n] = buf[:n]
        if n < frames:  # 块长不一致时补静音，而不是让 numpy 广播抛异常中断流
            outdata[n:] = 0
        self._free.append(buf)
        return True

    def clear(self) -> None:
        """丢弃全部待播块（启动/停止时调用）。"""
        while True:
            try:
                self._free.append(self._q.get_nowait())
            except queue.Empty:
                return

    def _acquire(self, shape: tuple[int, ...]) -> np.ndarray:
        try:
            buf = self._free.pop()
        except IndexError:
            return np.empty(shape, dtype=np.float32)
        # 块时长/通道数变了，旧缓冲尺寸对不上，直接换新的
        return buf if buf.shape == shape else np.empty(shape, dtype=np.float32)
